(** compression_rescoring_api_utils.ml | Author: Catherine Wong, Kevin Ellis. 

Utility and implementation functions for the API endpoints in compression_rescoring_api. This contains much of the deserialization/serialization utilities to send things via JSON objects.

This also implements the underlying compression functionality, and draws on the original LAPS/DreamCoder reference implementation in compression.ml.
**)

open Core

open Gc

open Physics
open Pregex
open Tower
open Utils
open Type
open Program
open Enumeration
open Task
open Grammar

(* open Eg *)
open Versions

let verbose_compression = ref false;; (** Debug flag for verbose compression **)

(* If this is true, then we collect and report data on the sizes of the version spaces, for each program, and also for each round of inverse beta *)
let collect_data = ref false;;


(** Standard hyperparameters for the compressor. **)
let kwargs_string = "kwargs";;
let max_candidates_per_compression_step_string = "max_candidates_per_compression_step";;
let max_grammar_candidates_to_retain_for_rewriting_string = "max_grammar_candidates_to_retain_for_rewriting";;
let max_compression_steps_string = "max_compression_steps";;
let arity_string = "arity";; 
let pseudocounts_string = "pseudocounts";;
let aic_string = "aic";;
let cpus_string = "cpus";;
let structure_penalty_string = "structure_penalty";;
let language_alignments_weight_string = "language_alignments_weight";;
let language_alignments_string = "language_alignments";;
let top_k_string = "top_k";;

(** Default values for hyperparameters for the compressor. **)
let default_max_grammar_candidates_to_retain_for_rewriting = 1;;
let default_max_candidates_per_compression_step = 200;;
let default_max_compression_steps = 1000;;
let default_arity = 3;;
let default_pseudocounts = 30.0;;
let default_structure_penalty = 1.5;;
let default_aic = 1.0;;
let default_cpus = 2;;
let default_language_alignments_weight = 0.0;;
let default_language_alignments = [];;
let default_top_k = 5;;

type alignment = {
  key : string;
  score : float;
  primitive_counts : (string*int) list;
};;

let deserialize_alignment j =
  (** Utility function: deserialize IBM-style alignment object.*) 
  let open Yojson.Basic.Util in
  let key = j |> member "key" |> to_string in
  let score = j |> member "score" |> to_float in 
  let primitive_counts = j |> member "primitive_counts" |> to_list |> List.map ~f: (fun j -> (
      j |> member "prim" |> to_string,
      j |> member "count" |> to_int
    ))
  in {key; score; primitive_counts};;

let deserialize_compressor_kwargs json_kwargs = 
  (** Utility function to deserialize common kwargs hyperparameters for the compression algorithm, or return defaults for each hyperparameter if not provided. **)
  let open Yojson.Basic.Util in
  let open Yojson.Basic in

  let max_candidates_per_compression_step = try
    json_kwargs |> member max_candidates_per_compression_step_string |> to_int
  with _ -> default_max_candidates_per_compression_step in

  let max_grammar_candidates_to_retain_for_rewriting = try
    json_kwargs |> member  max_grammar_candidates_to_retain_for_rewriting_string |> to_int
  with _ -> default_max_grammar_candidates_to_retain_for_rewriting in

  let max_compression_steps = try
    json_kwargs |> member max_compression_steps_string |> to_int
  with _ -> default_max_candidates_per_compression_step in

  let top_k = try
    json_kwargs |> member top_k_string |> to_int
  with _ -> default_top_k in

  let arity = try
    json_kwargs |> member arity_string |> to_int
  with _ -> default_arity in

  let pseudocounts = try
    json_kwargs |> member pseudocounts_string |> to_float
  with _ -> default_pseudocounts in

  let structure_penalty = try
    json_kwargs |> member structure_penalty_string |> to_float
  with _ -> default_structure_penalty in

  let aic = try
    json_kwargs |> member aic_string |> to_float
  with _ -> default_aic in

  let cpus =  try
    json_kwargs |> member cpus_string |> to_int
  with _ -> default_cpus in

  let language_alignments_weight = try
    json_kwargs |> member language_alignments_weight_string |> to_float
  with _ -> default_language_alignments_weight in

  let language_alignments = (try
    json_kwargs |> member language_alignments_string |> to_list 
  with _ -> []) in
  let language_alignments = language_alignments |> List.map ~f:deserialize_alignment in

  (** Log the KWARGS *)
  let () = (Printf.eprintf "[ocaml] kwarg: \t max_candidates_per_compression_step %d \n" (max_candidates_per_compression_step)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t max_grammar_candidates_to_retain_for_rewriting %d \n" (max_grammar_candidates_to_retain_for_rewriting)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t max_compression_steps %d \n" (max_compression_steps)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t top_k %d \n" (top_k)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t arity %d \n" (arity)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t pseudocounts %f \n" (pseudocounts)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t structure_penalty %f \n" (structure_penalty)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t aic %f \n" (aic)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t cpus %d \n" (cpus)) in

  let () = (Printf.eprintf "[ocaml] kwarg: \t language_alignments_weight %f \n" (language_alignments_weight)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t with %d language_alignments \n" (List.length language_alignments)) in
  
  max_candidates_per_compression_step, max_grammar_candidates_to_retain_for_rewriting, max_compression_steps, top_k, arity, pseudocounts, structure_penalty, aic, cpus, language_alignments_weight, language_alignments
  ;;


(** Compressor algorithm. Utility functions for compressing the grammar and rewriting frontiers. *)

(** Score = length of counts * score **)
let alignment_len_score alignment = 
  let just_counts =  alignment.primitive_counts |> List.map ~f: (fun (_, c) -> c) in
  let count_len = just_counts |> (List.fold_right ~f:(+) ~init:0) in
  let vocab_score = (List.length alignment.primitive_counts) in
  let len_score = count_len + vocab_score in 
  (Float.of_int len_score) *. alignment.score

let align_to_string alignment = 
  let _ = Printf.eprintf "Key: %s\n" alignment.key in
  let _ = Printf.eprintf "Score: %f\n" alignment.score in 
  let _ = alignment.primitive_counts |> List.map ~f: (
    fun (p, c) ->  Printf.eprintf "%s: %d | " p c
    ) in 
  let _ = Printf.eprintf "Len score: %f\n" (alignment_len_score alignment) in
  Printf.eprintf "\n";;

let primitive_counts_to_string counts = 
  let _ = counts |> List.map ~f: (
    fun (p, c) ->  Printf.eprintf "%s: %d | " p c
    ) in 
  Printf.eprintf "\n";;

let deserialize_alignment j = 
  let open Yojson.Basic.Util in
  let key = j |> member "key" |> to_string in
  let score = j |> member "score" |> to_float in 
  let primitive_counts = j |> member "primitive_counts" |> to_list |> List.map ~f: (fun j -> (
      j |> member "prim" |> to_string,
      j |> member "count" |> to_int
    ))
  in {key; score; primitive_counts};;

let rec count_tokens counts tokens =
  match tokens with 
    | [] -> counts
    | (token :: rest) -> 
    let new_counts = 
    if List.Assoc.mem counts ~equal:( = ) token 
    then
      counts |> List.map ~f: (fun (t, c) -> if t = token then (t, c+1) else (t, c))
    else
      (token, 1) :: counts
    in (count_tokens new_counts rest)

let to_primitive_counts e = 
  let normal = beta_normal_form ~reduceInventions:true e in 
  let tokens = ((left_order_tokens false []) normal) in 
  count_tokens [] tokens 

(** For new primitive [T0: C0, T1: C1 ...] we consider the maximum number of times the full new primitive could go into the old counts **)
let rewrite_counts new_counts old_counts = 
  let rec _rewrite_counts max_rewrites new_counts old_counts =  
    match new_counts with 
    | [] -> max_rewrites 
    | (new_token, c) :: rest ->
      if List.Assoc.mem old_counts ~equal:( = ) new_token 
      then
        let old_count =  List.Assoc.find_exn old_counts ~equal:(=) new_token in
        let rewrites =  old_count / c in
        let max_rewrites = if (rewrites < max_rewrites) then rewrites else max_rewrites in
        _rewrite_counts max_rewrites rest old_counts
      else 
        _rewrite_counts 0 rest old_counts 
  in 
  let max_rewrites = _rewrite_counts 10000 new_counts old_counts
  in 
  if (max_rewrites = 10000) then 0 else max_rewrites

let rewrite_counts_and_score num_rewrites new_counts alignment =
  let old_counts = alignment.primitive_counts in 
  let rewritten = old_counts |> List.map ~f: (fun (token, c) ->
    if (List.Assoc.mem new_counts ~equal:( = ) token)
    then
      let new_count =  List.Assoc.find_exn new_counts ~equal:(=) token in
      (token, c - (num_rewrites * new_count))
    else
      (token, c)
  ) in 
  let rewritten = ("invention", num_rewrites) :: rewritten in
  let rewritten_alignment = {key = alignment.key; score = alignment.score; primitive_counts = rewritten} in
  alignment_len_score rewritten_alignment

let rewrite_score new_counts alignment = 
  let max_rewrites = rewrite_counts new_counts alignment.primitive_counts in
  if max_rewrites > 0 
  then 
    rewrite_counts_and_score max_rewrites new_counts alignment 
  else
    alignment_len_score alignment

(** Calculates score for rewriting all alignments with new primitive.
  Logprobs are negative, so shorter MDL = less negative; so a 
  bigger score is better.
*)
let all_rewrite_alignments_score alignments new_primitive = 
  let new_counts = to_primitive_counts new_primitive in
  alignments |> List.map ~f: (fun alignment ->
    rewrite_score new_counts alignment) |> (List.fold_right ~f:(+.) ~init:0.0)

let all_initial_alignment_score alignments = 
  alignments |> List.map ~f: (fun alignment -> alignment_len_score alignment) |> (List.fold_right ~f:(+.) ~init:0.0)

let restrict ~topK g frontier =
  let restriction =
    frontier.programs |> List.map ~f:(fun (p,ll) ->
        (ll+.likelihood_under_grammar g frontier.request p,p,ll)) |>
    sort_by (fun (posterior,_,_) -> 0.-.posterior) |>
    List.map ~f:(fun (_,p,ll) -> (p,ll))
  in
  {request=frontier.request; programs=List.take restriction topK}


let inside_outside ~pseudoCounts g (frontiers : frontier list) =
  let summaries = frontiers |> List.map ~f:(fun f ->
      f.programs |> List.map ~f:(fun (p,l) ->
          let s = make_likelihood_summary g f.request p in
          (l, s))) in

  let update g =
    let weighted_summaries = summaries |> List.map ~f:(fun ss ->
        let log_weights = ss |> List.map ~f:(fun (l,s) ->
            l+. summary_likelihood g s) in
        let z = lse_list log_weights in
        List.map2_exn log_weights ss ~f:(fun lw (_,s) -> (exp (lw-.z),s))) |>
                             List.concat
    in

    let s = mix_summaries weighted_summaries in
    let possible p = Hashtbl.fold ~init:0. s.normalizer_frequency  ~f:(fun ~key ~data accumulator ->
        if List.mem ~equal:program_equal key p then accumulator+.data else accumulator)
    in
    let actual p = match Hashtbl.find s.use_frequency p with
      | None -> 0.
      | Some(f) -> f
    in

    {g with
     logVariable = log (actual (Index(0)) +. pseudoCounts) -. log (possible (Index(0)) +. pseudoCounts);
     library = g.library |> List.map ~f:(fun (p,t,_,u) ->
         let l = log (actual p +. pseudoCounts) -. log (possible p +. pseudoCounts) in
       (p,t,l,u))}
  in
  let g = update g in
  (g,
   summaries |> List.map ~f:(fun ss ->
     ss |> List.map ~f:(fun (l,s) -> l+. summary_likelihood g s) |> lse_list) |> fold1 (+.))
    
    
let grammar_induction_score ~aic ~structurePenalty ~pseudoCounts frontiers g =
  let g,ll = inside_outside ~pseudoCounts g frontiers in

  let production_size = function
    | Primitive(_,_,_) -> 1
    | Invented(_,e) -> begin 
        (* Ignore illusory fix1/abstraction, it does not contribute to searching cost *)
        let e = recursively_get_abstraction_body e in
        match e with
        | Apply(Apply(Primitive(_,"fix1",_),i),b) -> (assert (is_index i); program_size b)
        | _ -> program_size e
      end
    | _ -> raise (Failure "Element of grammar is neither primitive nor invented")
  in 

  (g,
   ll-. aic*.(List.length g.library |> Float.of_int) -.
   structurePenalty*.(g.library |> List.map ~f:(fun (p,_,_,_) ->
       production_size p) |> sum |> Float.of_int))


exception EtaExpandFailure;;

let eta_long request e =
  let context = ref empty_context in

  let make_long e request =
    if is_arrow request then Some(Abstraction(Apply(shift_free_variables 1 e, Index(0)))) else None
  in 

  let rec visit request environment e = match e with
    | Abstraction(b) when is_arrow request ->
      Abstraction(visit (right_of_arrow request) (left_of_arrow request :: environment) b)
    | Abstraction(_) -> raise EtaExpandFailure
    | _ -> match make_long e request with
      | Some(e') -> visit request environment e'
      | None -> (* match e with *)
        (* | Index(i) -> (unify' context request (List.nth_exn environment i); e) *)
        (* | Primitive(t,_,_) | Invented(t,_) -> *)
        (*   (let t = instantiate_type' context t in *)
        (*    unify' context t request; *)
        (*    e) *)
        (* | Abstraction(_) -> assert false *)
        (* | Apply(_,_) -> *)
        let f,xs = application_parse e in
        let ft = match f with
          | Index(i) -> environment $$ i |> applyContext' context
          | Primitive(t,_,_) | Invented(t,_) -> instantiate_type' context t
          | Abstraction(_) -> assert false (* not in beta long form *)
          | Apply(_,_) -> assert false
        in
        unify' context request (return_of_type ft);
        let ft = applyContext' context ft in
        let xt = arguments_of_type ft in
        if List.length xs <> List.length xt then raise EtaExpandFailure else
          let xs' =
            List.map2_exn xs xt ~f:(fun x t -> visit (applyContext' context t) environment x)
          in
          List.fold_left xs' ~init:f ~f:(fun return_value x ->
              Apply(return_value,x))
  in

  let e' = visit request [] e in
  
  assert (tp_eq
            (e |> closed_inference |> canonical_type)
            (e' |> closed_inference |> canonical_type));
  e'
;;

let normalize_invention i =
  (* Raises UnificationFailure if i is not well typed *)
  let mapping = free_variables i |> List.dedup_and_sort ~compare:(-) |> List.mapi ~f:(fun i v -> (v,i)) in

  let rec visit d = function
    | Index(i) when i < d -> Index(i)
    | Index(i) -> Index(d + List.Assoc.find_exn ~equal:(=) mapping (i - d))
    | Abstraction(b) -> Abstraction(visit (d + 1) b)
    | Apply(f,x) -> Apply(visit d f,
                          visit d x)
    | Primitive(_,_,_) | Invented(_,_) as e -> e
  in
  
  let renamed = visit 0 i in
  let abstracted = List.fold_right mapping ~init:renamed ~f:(fun _ e -> Abstraction(e)) in
  make_invention abstracted
    

let rewrite_with_invention i =
  (* Raises EtaExpandFailure if this is not successful *)
  let mapping = free_variables i |> List.dedup_and_sort ~compare:(-) |> List.mapi ~f:(fun i v -> (i,v)) in
  let closed = normalize_invention i in
  (* FIXME : no idea whether I got this correct or not... *)
  let applied_invention = List.fold_left ~init:closed
      (List.range ~start:`exclusive ~stop:`inclusive ~stride:(-1) (List.length mapping) 0)
      ~f:(fun e i -> Apply(e,Index(List.Assoc.find_exn ~equal:(=) mapping i)))
  in

  let rec visit e =
    if program_equal e i then applied_invention else
      match e with
      | Apply(f,x) -> Apply(visit f, visit x)
      | Abstraction(b) -> Abstraction(visit b)
      | Index(_) | Primitive(_,_,_) | Invented(_,_) -> e
  in
  fun request e ->
    try
      let e' = visit e |> eta_long request in
      assert (program_equal
                (beta_normal_form ~reduceInventions:true e)
                (beta_normal_form ~reduceInventions:true e'));
      e'
    with UnificationFailure -> begin
        if !verbose_compression then begin  
          Printf.eprintf "WARNING: rewriting with invention gave ill typed term.\n";
          Printf.eprintf "Original:\t\t%s\n" (e |> string_of_program);
          Printf.eprintf "Original:\t\t%s\n" (e |> beta_normal_form ~reduceInventions:true |> string_of_program);
          Printf.eprintf "Rewritten:\t\t%s\n" (visit e |> string_of_program);
          Printf.eprintf "Rewritten:\t\t%s\n" (visit e |> beta_normal_form ~reduceInventions:true |> string_of_program);
          Printf.eprintf "Going to proceed as if the rewrite had failed - but look into this because it could be a bug.\n";
          flush_everything()
        end;
        let normal_original = e |> beta_normal_form ~reduceInventions:true in
        let normal_rewritten = e |> visit |> beta_normal_form ~reduceInventions:true in
        assert (program_equal normal_original normal_rewritten);        
        raise EtaExpandFailure
    end
      

let nontrivial e =
  let indices = ref [] in
  let duplicated_indices = ref 0 in
  let primitives = ref 0 in
  let rec visit d = function
    | Index(i) ->
      let i = i - d in
      if List.mem ~equal:(=) !indices i
      then incr duplicated_indices
      else indices := i :: !indices
    | Apply(f,x) -> (visit d f; visit d x)
    | Abstraction(b) -> visit (d + 1) b
    | Primitive(_,_,_) | Invented(_,_) -> incr primitives
  in
  visit 0 e;
  !primitives > 1 || !primitives = 1 && !duplicated_indices > 0
;;

open Zmq

type worker_command =
  | Rewrite of program list
  | RewriteEntireFrontiers of program
  | KillWorker
  | FinalFrontier of program
  | BatchedRewrite of program list
    
let compression_worker connection ~inline ~arity ~bs ~topK g frontiers test_frontiers =
  let context = Zmq.Context.create() in
  let socket = Zmq.Socket.create context Zmq.Socket.req in
  Zmq.Socket.connect socket connection;
  let send data = Zmq.Socket.send socket (Marshal.to_string data []) in
  let receive() = Marshal.from_string (Zmq.Socket.recv socket) 0 in


  let original_frontiers = frontiers in
  let frontiers = ref (List.map ~f:(restrict ~topK g) frontiers) in

  let v = new_version_table() in

  (* calculate candidates from the frontiers we can see *)
  let frontier_indices : int list list = time_it ~verbose:!verbose_compression
      "(worker) calculated version spaces" (fun () ->
      !frontiers |> List.map ~f:(fun f -> f.programs |> List.map ~f:(fun (p,_) ->
              incorporate v p |> n_step_inversion v ~inline ~n:arity))) in
  if !collect_data then begin
    List.iter2_exn !frontiers frontier_indices ~f:(fun frontier indices ->
        List.iter2_exn (frontier.programs) indices ~f:(fun (p,_) index ->
            let rec program_size = function
              | Apply(f,x) -> 1 + program_size f + program_size x
              | Abstraction(b) -> 1 + program_size b
              | Index(_) | Invented(_,_) | Primitive(_,_,_) -> 1
            in
            let rec program_height = function
              | Apply(f,x) -> 1 + (max (program_height f) (program_height x))
              | Abstraction(b) -> 1 + program_height b
              | Index(_) | Invented(_,_) | Primitive(_,_,_) -> 1
            in
            Printf.eprintf "DATA\t%s\tsize=%d\theight=%d\t|vs|=%d\t|[vs]|=%f\n" (string_of_program p)
              (program_size p) (program_height p)
              (reachable_versions v [index] |> List.length)
              (log_version_size v index)
          ))
  end;
  if !verbose_compression then
    Printf.eprintf "(worker) %d distinct version spaces enumerated; %d accessible vs size; vs log sizes: %s\n"
      v.i2s.ra_occupancy
      (frontier_indices |> List.concat |> reachable_versions v |> List.length)
      (frontier_indices |> List.concat |> List.map ~f:(Float.to_string % log_version_size v)
       |> join ~separator:"; ");

  let v, frontier_indices = garbage_collect_versions ~verbose:!verbose_compression v frontier_indices in
  Gc.compact();
  
  let cost_table = empty_cost_table v in

  (* pack the candidates into a version space for efficiency *)
  let candidate_table = new_version_table() in
  let candidates : int list list = time_it ~verbose:!verbose_compression "(worker) proposed candidates"
      (fun () ->
      let reachable : int list list = frontier_indices |> List.map ~f:(reachable_versions v) in
      let inhabitants : int list list = reachable |> List.map ~f:(fun indices ->
          List.concat_map ~f:(snd % minimum_cost_inhabitants cost_table) indices |>
          List.dedup_and_sort ~compare:(-) |> 
          List.map ~f:(List.hd_exn % extract v) |>
          List.filter ~f:nontrivial |>
          List.map ~f:(incorporate candidate_table)) in 
          inhabitants)
  in
  if !verbose_compression then Printf.eprintf "(worker) Total candidates: [%s] = %d, packs into %d vs\n"
      (candidates |> List.map ~f:(Printf.sprintf "%d" % List.length) |> join ~separator:";")
      (candidates |> List.map ~f:List.length |> sum)
      (candidate_table.i2s.ra_occupancy);
  flush_everything();

  (* relay this information to the master, whose job it is to pool the candidates *)
  send (candidates,candidate_table.i2s);
  let candidate_table = () in
  let candidates : program list = receive() in
  let candidates : int list = candidates |> List.map ~f:(incorporate v) in
  
  if !verbose_compression then
    (Printf.eprintf "(worker) Got %d candidates.\n" (List.length candidates);
     flush_everything());

  let candidate_scores : float list = time_it ~verbose:!verbose_compression "(worker) beamed version spaces"
      (fun () ->
      beam_costs' ~ct:cost_table ~bs candidates frontier_indices)
  in

  send candidate_scores;
   (* I hope that this leads to garbage collection *)
  let candidate_scores = ()
  and cost_table = ()
  in
  Gc.compact();

  let rewrite_frontiers invention_source =
    time_it ~verbose:!verbose_compression "(worker) rewrote frontiers" (fun () -> 
        time_it ~verbose:!verbose_compression "(worker) gc during rewrite" Gc.compact;
        let intersectionTable = Some(Hashtbl.Poly.create()) in
        let i = incorporate v invention_source in
        let rewriter = rewrite_with_invention invention_source in
        (* Extract the frontiers in terms of the new primitive *)
        let new_cost_table = empty_cheap_cost_table v in
        let new_frontiers = List.map !frontiers
            ~f:(fun frontier ->
                let programs' =
                  List.map frontier.programs ~f:(fun (originalProgram, ll) ->
                      let index = incorporate v originalProgram |> n_step_inversion ~inline v ~n:arity in
                      let program =
                        minimal_inhabitant ~intersectionTable new_cost_table ~given:(Some(i)) index |> get_some
                      in 
                      let program' =
                        try rewriter frontier.request program
                        with EtaExpandFailure -> originalProgram
                      in
                      (program',ll))
                in 
                {request=frontier.request;
                 programs=programs'})
        in
        new_frontiers)
  in

  let batched_rewrite inventions =
    time_it ~verbose:!verbose_compression "(worker) batched rewrote frontiers" (fun () ->
        Gc.compact();
        let invention_indices : int list = inventions |> List.map ~f:(incorporate v) in
        let frontier_indices : int list list =
          !frontiers |> List.map ~f:(fun f ->
              f.programs |> List.map ~f:(n_step_inversion ~inline v ~n:arity % incorporate v % fst))
        in
        clear_dynamic_programming_tables v;
        let refactored = batched_refactor ~ct:(empty_cost_table v) invention_indices frontier_indices in
        Gc.compact();
        List.map2_exn refactored inventions ~f:(fun new_programs invention ->
            let rewriter = rewrite_with_invention invention in
            List.map2_exn new_programs !frontiers ~f:(fun new_programs frontier ->
                let programs' =
                  List.map2_exn new_programs frontier.programs ~f:(fun program (originalProgram, ll) ->
                      if not (program_equal
                                (beta_normal_form ~reduceInventions:true program)
                                (beta_normal_form ~reduceInventions:true originalProgram)) then
                        (Printf.eprintf "FATAL: %s refactored into %s\n"
                           (string_of_program originalProgram)
                           (string_of_program program);
                         Printf.eprintf "This has never occurred before! Definitely send this to Kevin, if this occurs it is a terrifying bug.\n";
                        assert (false));
                      let program' =
                        try rewriter frontier.request program
                        with EtaExpandFailure -> originalProgram
                      in
                      (program',ll))
                in 
                {request=frontier.request;
                 programs=programs'})))
  in

  let final_rewrite invention =
    (* As our last act, free as much memory as we can *)
    deallocate_versions v; Gc.compact();

    (* exchanging time for memory - invert everything again *)
    let train_frontiers_len = List.length !frontiers in 
    frontiers := (original_frontiers @ test_frontiers);
    let v = new_version_table() in
    let frontier_inversions = Hashtbl.Poly.create() in
    time_it ~verbose:!verbose_compression "(worker) did final inversion" (fun () -> 
        !frontiers |> List.iter ~f:(fun f ->
            f.programs |> List.iter ~f:(fun (p,_) ->
                Hashtbl.set frontier_inversions
                  ~key:(incorporate v p)
                  ~data:(n_step_inversion ~inline v ~n:arity (incorporate v p)))));
    clear_dynamic_programming_tables v; Gc.compact();    
    
    let i = incorporate v invention in
    let new_cost_table = empty_cheap_cost_table v in
    time_it ~verbose:!verbose_compression "(worker) did final refactor" (fun () -> 
        let final_refactored_combined_frontiers = List.map !frontiers
          ~f:(fun frontier ->
              let programs' =
                List.map frontier.programs ~f:(fun (originalProgram, ll) ->
                    let index = Hashtbl.find_exn frontier_inversions (incorporate v originalProgram) in 
                    let program =
                      minimal_inhabitant new_cost_table ~given:(Some(i)) index |> get_some
                    in 
                    let program' =
                      try rewrite_with_invention invention frontier.request program
                      with EtaExpandFailure -> originalProgram
                    in
                    (program',ll))
              in 
              {request=frontier.request;
               programs=programs'})
            in 
            let (refactored_final_train, refactored_final_test) = List.split_n final_refactored_combined_frontiers train_frontiers_len in 
            [(refactored_final_train, refactored_final_test)]
      )
      in

  while true do
    match receive() with
    | Rewrite(i) -> send (i |> List.map ~f:rewrite_frontiers)
    | RewriteEntireFrontiers(i) ->
      (frontiers := original_frontiers;
       send (rewrite_frontiers i))
    | BatchedRewrite(inventions) -> send (batched_rewrite inventions)
    | FinalFrontier(invention) ->
      (frontiers := original_frontiers;
       send (final_rewrite invention);
       Gc.compact())
    | KillWorker -> 
       (Zmq.Socket.close socket;
        Zmq.Context.terminate context;
       exit 0)
  done;;

let compression_step_master ~inline ~nc ~structurePenalty ~aic ~pseudoCounts ~lc_score ?arity:(arity=3) ~bs ~topI ~topK ?max_grammar_candidates_to_retain_for_rewriting:(max_grammar_candidates_to_retain_for_rewriting=1) g frontiers test_frontiers language_alignments =
  (**
    params: 
      retain_all_grammar_candidates: if True, returns the topI grammar candidates and rewrites frontiers with respect to them.
  *)
  let sockets = ref [] in
  let timestamp = Time.now() |> Time.to_filename_string ~zone:Time.Zone.utc in
  let fork_worker (train_frontiers, test_frontiers) =
    let p = List.length !sockets in
    let address = Printf.sprintf "ipc:///tmp/compression_ipc_%s_%d" timestamp p in
    sockets := !sockets @ [address];

    match Unix.fork() with
    | `In_the_child -> compression_worker address ~arity ~bs ~topK ~inline g train_frontiers test_frontiers
    | _ -> ()
  in

  if !verbose_compression then ignore(Unix.system "ps aux|grep compression 1>&2");

  (** TODO: CATWONG: need to divide these correctly amongst the different groups -- right now, not guaranteed for frontiers and test_frontiers to be the same size.*)
  let divide_work_fairly nc xs =
    let nt = List.length xs in
    let base_count = nt/nc in
    let residual = nt - base_count*nc in
    let rec partition residual xs =
      let this_count =
        base_count + (if residual > 0 then 1 else 0)
      in
      match xs with
      | [] -> []
      | _ :: _ ->
        let prefix, suffix = List.split_n xs this_count in
        prefix :: partition (residual - 1) suffix
    in
    partition residual xs
  in
  let start_time = Time.now () in
  let partitioned_train = divide_work_fairly nc frontiers in
  let partitioned_test = divide_work_fairly nc test_frontiers in 
  let partitioned_train, partitioned_test = pad_to_equal_length_2 partitioned_train partitioned_test ~pad:[] in
  List.zip_exn partitioned_train partitioned_test |> List.iter ~f:fork_worker;

  (* Now that we have created the workers, we can make our own sockets *)
  let context = Zmq.Context.create() in
  let sockets = !sockets |> List.map ~f:(fun address ->
      let socket = Zmq.Socket.create context Zmq.Socket.rep in
      Zmq.Socket.bind socket address;
      socket)
  in
  let send data =
    let data = Marshal.to_string data [] in
    sockets |> List.iter ~f:(fun socket -> Zmq.Socket.send socket data)
  in
  let receive socket = Marshal.from_string (Zmq.Socket.recv socket) 0 in
  let finish() =
    send KillWorker;
    sockets |> List.iter ~f:(fun s -> Zmq.Socket.close s);
    Zmq.Context.terminate context
  in 
    
  
  let candidates : program list list = sockets |> List.map ~f:(fun s ->
      let candidate_message : (int list list)*(vs ra) = receive s in
      let (candidates, candidate_table) = candidate_message in
      let candidate_table = {(new_version_table()) with i2s=candidate_table} in
      candidates |> List.map ~f:(List.map ~f:(singleton_head % extract candidate_table))) |> List.concat in  
  let candidates : program list = occurs_multiple_times (List.concat candidates) in
  Printf.eprintf "Total number of candidates: %d\n" (List.length candidates);
  Printf.eprintf "Constructed version spaces and coalesced candidates in %s.\n"
    (Time.diff (Time.now ()) start_time |> Time.Span.to_string);
  flush_everything();
  
  send candidates;

  let candidate_scores : float list list =
    sockets |> List.map ~f:(fun s -> let ss : float list = receive s in ss)
  in
  if !verbose_compression then (Printf.eprintf "(master) Received worker beams\n"; flush_everything());
  let candidates : program list = 
    candidate_scores |> List.transpose_exn |>
    List.map ~f:(fold1 (+.)) |> List.zip_exn candidates |>
    List.sort ~compare:(fun (_,s1) (_,s2) -> Float.compare s1 s2) |> List.map ~f:fst
  in
  let candidates = List.take candidates topI in
  let candidates = candidates |> List.filter ~f:(fun candidate ->
      try
        let candidate = normalize_invention candidate in
        not (List.mem ~equal:program_equal (grammar_primitives g) candidate)
      with UnificationFailure -> false) (* not well typed *)
  in
  Printf.eprintf "Trimmed down the beam, have only %d best candidates\n"
    (List.length candidates);
  flush_everything();

  match candidates with
  | [] -> (finish(); None)
  | _ -> 

  (* now we have our final list of candidates! *)
  (* ask each of the workers to rewrite w/ each candidate *)
  send @@ BatchedRewrite(candidates);
  (* For each invention, the full rewritten frontiers *)
  let new_frontiers : frontier list list =
    time_it "Rewrote topK" (fun () ->
        sockets |> List.map ~f:receive |> List.transpose_exn |> List.map ~f:List.concat)
  in
  assert (List.length new_frontiers = List.length candidates);
  
  (** Compression score: globally score the topI candidates.*)
  let score frontiers candidate =
    let new_grammar = uniform_grammar (normalize_invention candidate :: grammar_primitives g) in
    let g',s = grammar_induction_score ~aic ~pseudoCounts ~structurePenalty frontiers new_grammar in
    if !verbose_compression then
      (let source = normalize_invention candidate in
       Printf.eprintf "Invention %s : %s\n\tContinuous score %f\n"
         (string_of_program source)
         (closed_inference source |> string_of_type)
         s;
       frontiers |> List.iter ~f:(fun f -> Printf.eprintf "%s\n" (string_of_frontier f));
       Printf.eprintf "\n"; flush_everything());
    (g',s)
  in 
  let _,initial_mdl_score = grammar_induction_score ~aic ~structurePenalty ~pseudoCounts
      (frontiers |> List.map ~f:(restrict ~topK g)) g
  in
  Printf.eprintf "Initial score: %f\n" initial_mdl_score;
  
  let language_score language_alignments candidate =  all_rewrite_alignments_score language_alignments candidate
  in 
  let initial_language_score = all_initial_alignment_score language_alignments in 
  Printf.eprintf "Initial language score: %f\n" initial_language_score;
  
  let initial_joint_score = ((1.0 -. lc_score) *. initial_mdl_score) +. (lc_score *. initial_language_score) in
  Printf.eprintf "Initial joint score: %f using LC of %f\n" initial_joint_score lc_score;

  let _ = Printf.eprintf "Here in the master worker: we still have %d language alignments;\n" (List.length language_alignments) in 
  let (g',best_mdl_score), best_mdl_candidate = time_it "Scored candidates" (fun () ->
      List.map2_exn candidates new_frontiers ~f:(fun candidate frontiers ->
          (score frontiers candidate, candidate)) |> minimum_by (fun ((_,s),_) -> -.s))
  in
  let _ = Printf.eprintf "Best MDL score: %f with %s\n" best_mdl_score (string_of_program best_mdl_candidate) in
  if best_mdl_score < initial_mdl_score then
      (Printf.eprintf "No improvement possible with MDL.\n");
  
  (** Combine the scores and retain the best candidates for globally rewritting. **)
  let best_scores_and_candidates 
  = time_it "Scored candidates with language and grammar" (fun () ->
      List.map2_exn candidates new_frontiers ~f:(fun candidate frontiers ->
          let (new_g, mdl_score) = score frontiers candidate in
          let lang_score = language_score language_alignments candidate in
          let combined_score = ((1.0 -. lc_score) *. mdl_score) +. (lc_score *. lang_score) in
          ((new_g, combined_score, mdl_score), candidate))
          |> sort_by (fun ((_,s,_),_) -> -.s)
          |> slice 0 max_grammar_candidates_to_retain_for_rewriting)
      in
  let _ = Printf.eprintf "Retained top %d candidates for rewriting \n" (List.length best_scores_and_candidates)  in
  let (g',best_joint_score, best_mdl_score), best_candidate = best_scores_and_candidates |> List.hd_exn 
  in
  let _ = Printf.eprintf "Best joint score: %f with %s\n" best_joint_score (string_of_program best_candidate) in
  
  if best_joint_score < initial_joint_score then
      (Printf.eprintf "No improvement possible with the best joint score.\n"; finish(); None)
    else
      (let new_primitive = grammar_primitives g' |> List.hd_exn in
       Printf.eprintf "Improved score to %f (dScore=%f) w/ new primitive\n\t%s : %s\n"
         best_joint_score (best_joint_score-.initial_joint_score)
         (string_of_program new_primitive) (closed_inference new_primitive |> canonical_type |> string_of_type);
       flush_everything();
       

      (* Rewrite all of the remaining candidates. *)
      let grammar_candidates, compression_scores_for_candidates, train_frontiers_for_candidates, test_frontiers_for_candidates = ref [], ref [], ref [], ref [] in
      let _ = time_it "Rewrote all frontiers for all candidates" (
        fun () ->
          List.map best_scores_and_candidates ~f:(
            fun ((g_candidate', combined_score, mdl_score), invention_candidate) ->
            let combined_frontiers'' : (frontier list * frontier list) list = time_it "rewrote all of the frontiers for one candidate" (fun () ->
             let _ = (Printf.eprintf "Rewriting with new primitive\n\t%s : %s\n" (string_of_program invention_candidate) (closed_inference invention_candidate |> canonical_type |> string_of_type)) in 
              send @@ FinalFrontier(invention_candidate);
              sockets |> List.map ~f:receive |> List.concat)
              in
              let train_frontiers'' = combined_frontiers'' |> List.map ~f:fst |> List.concat in 
              let test_frontiers'' = combined_frontiers'' |> List.map ~f:snd |> List.concat in 
              let g'' = inside_outside ~pseudoCounts g_candidate' train_frontiers'' |> fst in
              
              grammar_candidates := !grammar_candidates @ [g''];
              compression_scores_for_candidates := !compression_scores_for_candidates @ [combined_score];
              train_frontiers_for_candidates := !train_frontiers_for_candidates @ [train_frontiers''];
              test_frontiers_for_candidates := !test_frontiers_for_candidates @ [test_frontiers'']; 
          )
        )
       in 
       finish();
       let _ = (Printf.eprintf "Returning rewritten grammars and frontiers for %d candidates\n" (List.length !grammar_candidates)) in 
       Some(!grammar_candidates, !compression_scores_for_candidates,!train_frontiers_for_candidates, !test_frontiers_for_candidates))
      ;;
let find_new_primitive old_grammar new_grammar =
  (** Helper sub-function for reporting new primitives in a grammar with respect to another grammar. *)
  new_grammar |> grammar_primitives |> List.filter ~f:(fun p ->
      not (List.mem ~equal:program_equal (old_grammar |> grammar_primitives) p)) |>
  singleton_head;;

let illustrate_new_primitive new_grammar primitive frontiers =
  (** Helper sub-function for reporting where primitives are used in a set of frontiers. *)
  let illustrations = 
    frontiers |> List.filter_map ~f:(fun frontier ->
        let best_program = (restrict ~topK:1 new_grammar frontier).programs |> List.hd_exn |> fst in
        if List.mem ~equal:program_equal (program_subexpressions best_program) primitive then
          Some(best_program)
        else None)
  in
  Printf.eprintf "[ocaml] New primitive is used %d times in the best programs in each of the training frontiers.\n"
      (List.length illustrations);
    Printf.eprintf "[ocaml] Here is where it is used:\n";
    illustrations |> List.iter ~f:(fun program -> Printf.eprintf "  %s\n" (string_of_program program));;

let compress_grammar_candidates_and_rewrite_frontiers_for_each ~grammar ~train_frontiers ~test_frontiers ?language_alignments:(language_alignments=[]) ~max_candidates_per_compression_step
~max_grammar_candidates_to_retain_for_rewriting ~max_compression_steps ~top_k ~arity ~pseudocounts ~structure_penalty ~aic ~cpus  ?language_alignments_weight:(language_alignments_weight=0.0) = 
  (** compress_grammar_candidates_and_rewrite_frontiers_for_each: compresses the grammar with respect to train_frontiers to retain max_candidates_per_compression_step top candidates, and rewrites train/test frontiesrs *with respect to each candidate grammar.*

  :params:
       See: compress_grammar_and_rewrite_frontiers_loop or compression_step_master. Note that this does NOT have a max_compression_steps, as we currently return candidates after a single step.
  *)
  let compress_rewrite_step = compression_step_master ~nc:cpus in
  match time_it "[ocaml] Completed one iteration of compression and rewriting for all candidates."
              (fun () -> compress_rewrite_step ~inline:true ~structurePenalty:structure_penalty ~aic ~pseudoCounts:pseudocounts ~lc_score:language_alignments_weight ~arity ~bs:100000 ~topI:max_candidates_per_compression_step ~topK:top_k grammar train_frontiers test_frontiers language_alignments
              ~max_grammar_candidates_to_retain_for_rewriting)
      with
      | None -> [grammar], [0.0], [train_frontiers], [test_frontiers]
      | Some(grammar_candidates', compression_scores_candidates', train_frontiers_candidates', test_frontiers_candidates') -> 
        grammar_candidates', compression_scores_candidates',  train_frontiers_candidates', test_frontiers_candidates'


let compress_grammar_and_rewrite_frontiers_loop 
~grammar ~train_frontiers ~test_frontiers ?language_alignments:(language_alignments=[]) ~max_candidates_per_compression_step ~max_compression_steps ~top_k ~arity ~pseudocounts ~structure_penalty ~aic ~cpus  ?language_alignments_weight:(language_alignments_weight=0.0) = 
(** compress_grammar_and_rewrite_frontiers_loop: combined utility function that compresses the grammar with respect to the train_frontiers AND rewrites train/test frontiers under the new grammar.
Compare to the implementation in compression.ml
:params:
  grammar: Grammar object.
  train_frontiers, test_frontiers: List of Frontier objects.
  language_alignments: List of alignment objects (for LAPS-style language-guided compression.)

  max_candidates_per_compression_step: max beam size of library candidate functions to consider at each step.
  max_compressions_steps: maximum number of iterations to run the compressor.
  top_k: restrict compression to the top-k programs in the frontier.
  arity: maximum arity of the library candidates to consider.
  pseudocounts: pseudocount parameter for observing each existing primitive to smooth compression.
  structure_penalty: penalty weighting string over the size of the new function primitive.
  AIC: AIC weighting string over the description length of the library with the new primitive.
  CPUS: how many CPUS on which to run compression. If 1, serial. If multiple, parallelize.
  Language alignments weight: weighting string for the language alignment compression score.

:ret:
  compressed_grammars: compressed grammar with new DSL inventions.
  rewritten_train_frontiers, rewritten_test_frontiers: frontiers rewritten under the new DSL.
*)
  let compress_rewrite_step = compression_step_master ~nc:cpus in
  let rec compress_rewrite_report_loop ~max_compression_steps grammar train_frontiers test_frontiers =
    if max_compression_steps < 1 then
      (Printf.eprintf "[ocaml] Maximum compression steps reached; exiting compression.\n"; grammar, train_frontiers, test_frontiers)
    else
      match time_it "[ocaml] Completed one iteration of compression and rewriting."
              (fun () -> compress_rewrite_step 
              ~max_grammar_candidates_to_retain_for_rewriting:1
              ~inline:true ~structurePenalty:structure_penalty ~aic ~pseudoCounts:pseudocounts ~lc_score:language_alignments_weight ~arity ~bs:100000 ~topI:max_candidates_per_compression_step 
              ~topK:top_k grammar train_frontiers test_frontiers language_alignments)
      with
      | None -> grammar, train_frontiers, test_frontiers
      | Some(grammar_candidates', _, train_frontiers_candidates', test_frontiers_candidates') ->
        let (grammar', train_frontiers', test_frontiers') = List.hd_exn grammar_candidates', List.hd_exn train_frontiers_candidates', List.hd_exn test_frontiers_candidates' in
        illustrate_new_primitive grammar' (find_new_primitive grammar grammar') train_frontiers';
        flush_everything();
        compress_rewrite_report_loop (max_compression_steps - 1) grammar' train_frontiers' test_frontiers'
  in
  time_it "[ocaml] Completed ocaml compression." (fun () ->
    compress_rewrite_report_loop ~max_compression_steps grammar train_frontiers test_frontiers)
;;

