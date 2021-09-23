(** compression_rescoring_api_utils.ml | Author: Catherine Wong, Kevin Ellis. 

Utility functions for the API.
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

let verbose_compression = ref true;; (** Debug flag for verbose compression **)


(** Standard hyperparameters for the compressor. **)
let kwargs_string = "kwargs";;
let max_candidates_per_compression_step_string = "max_candidates_per_compression_step";;
let max_compression_steps_string = "max_compression_steps";;
let arity_string = "arity";; 
let pseudocounts_string = "pseudocounts";;
let aic_string = "aic";;
let cpus_string = "cpus";;
let structure_penalty_string = "structure_penalty";;
let language_alignments_weight_string = "language_alignments_weight";;
let language_alignments_string = "language_alignments";;

(** Default values for hyperparameters for the compressor. **)
let default_max_candidates_per_compression_step = 200;;
let default_max_compression_steps = 1000;;
let default_arity = 3;;
let default_pseudocounts = 30.0;;
let default_structure_penalty = 1.5;;
let default_aic = 1.0;;
let default_cpus = 1;;
let default_language_alignments_weight = 0.0;;
let default_language_alignments = [];;

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

  let max_compression_steps = try
    json_kwargs |> member max_compression_steps_string |> to_int
  with _ -> default_max_candidates_per_compression_step in

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
  let () = (Printf.eprintf "[ocaml] kwarg: \t max_compression_steps %d \n" (max_compression_steps)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t arity %d \n" (arity)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t pseudocounts %f \n" (pseudocounts)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t structure_penalty %f \n" (structure_penalty)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t aic %f \n" (aic)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t cpus %d \n" (cpus)) in

  let () = (Printf.eprintf "[ocaml] kwarg: \t language_alignments_weight %f \n" (language_alignments_weight)) in
  let () = (Printf.eprintf "[ocaml] kwarg: \t with %d language_alignments \n" (List.length language_alignments)) in
  
  max_candidates_per_compression_step, max_compression_steps, arity, pseudocounts, structure_penalty, aic, cpus, language_alignments_weight, language_alignments
  ;;


(** Compressor algorithm. Utility functions for compressing the grammar and rewriting frontiers. *)

let restrict ~topK g frontier =
  (* Utilty function for taking the topK likelihood programs in a frontier *)
  let restriction =
    frontier.programs |> List.map ~f:(fun (p,ll) ->
        (ll+.likelihood_under_grammar g frontier.request p,p,ll)) |>
    sort_by (fun (posterior,_,_) -> 0.-.posterior) |>
    List.map ~f:(fun (_,p,ll) -> (p,ll))
  in
  {request=frontier.request; programs=List.take restriction topK}

let serial_compress_rewrite_step ~grammar ~train_frontiers ~test_frontiers ~language_alignments ~max_candidates_per_compression_step ~max_compression_steps ~arity ~pseudocounts ~structure_penalty ~aic ~cpus  ~language_alignments_weight = 
  (** Ideally this would keep around the beam of max_candidates if max_compression_steps is < 1 *)
  None

let parallel_compress_rewrite_step_master ~cpus ~grammar ~train_frontiers ~test_frontiers ~language_alignments ~max_candidates_per_compression_step ~max_compression_steps ~arity ~pseudocounts ~structure_penalty ~aic ~cpus  ~language_alignments_weight = 
(** Ideally this would keep around the beam of max_candidates if max_compression_steps is < 1 *)
None

let get_parallel_or_serial_compress_rewrite_step ~cpus =
  (** Wrapper function: instantiates a parallel step if CPUS > 1 else gets the serial compression step. *) 
  let compress_rewrite_step = if cpus = 1 then serial_compress_rewrite_step else parallel_compress_rewrite_step_master ~cpus
in compress_rewrite_step

let compress_grammar_and_rewrite_frontiers_loop 
~grammar ~train_frontiers ~test_frontiers ?language_alignments:(language_alignments=[]) ~max_candidates_per_compression_step ~max_compression_steps ~arity ~pseudocounts ~structure_penalty ~aic ~cpus  ?language_alignments_weight:(language_alignments_weight=0.0) = 
(** compress_grammar_and_rewrite_frontiers_loop: combined utility function that compresses the grammar with respect to the train_frontiers AND rewrites train/test frontiers under the new grammar.
Compare to the implementation in compression.ml
:params:
  grammar: Grammar object.
  train_frontiers, test_frontiers: List of Frontier objects.
  language_alignments: List of alignment objects (for LAPS-style language-guided compression.)

  max_candidates_per_compression_step: max beam size of library candidate functions to consider at each step.
  max_compressions_steps: maximum number of iterations to run the compressor.
  arity: maximum arity of the library candidates to consider.
  pseudocounts: pseudocount parameter for observing each existing primitive to smooth compression.
  structure_penalty: penalty weighting string over the size of the new function primitive.
  AIC: AIC weighting string over the description length of the library with the new primitive.
  CPUS: how many CPUS on which to run compression. If 1, serial. If multiple, parallelize.
  Language alignments weight: weighting string for the language alignment compression score.

:ret:
  compressed_grammar: compressed grammar with new DSL inventions.
  rewritten_train_frontiers, rewritten_test_frontiers: frontiers rewritten under the new DSL.
*)

(* Helper sub functions for reporting any new primitives added during compression *)
let find_new_primitive old_grammar new_grammar =
  new_grammar |> grammar_primitives |> List.filter ~f:(fun p ->
      not (List.mem ~equal:program_equal (old_grammar |> grammar_primitives) p)) |>
  singleton_head
in
let illustrate_new_primitive new_grammar primitive frontiers =
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
    illustrations |> List.iter ~f:(fun program -> Printf.eprintf "  %s\n" (string_of_program program))
  in 

  let compress_rewrite_step = get_parallel_or_serial_compress_rewrite_step ~cpus in
  let rec compress_rewrite_report_loop ~max_compression_steps grammar train_frontiers test_frontiers =
    if max_compression_steps < 1 then
      (Printf.eprintf "[ocaml] Maximum compression steps reached; exiting compression.\n"; grammar, train_frontiers, test_frontiers)
    else
      match time_it "[ocaml] Completed one iteration of compression and rewriting."
              (fun () -> compress_rewrite_step ~grammar ~train_frontiers ~test_frontiers ~language_alignments ~max_candidates_per_compression_step ~max_compression_steps ~arity ~pseudocounts ~structure_penalty ~aic ~cpus  ~language_alignments_weight)
      with
      | None -> grammar, train_frontiers, test_frontiers
      | Some(grammar', train_frontiers', test_frontiers') ->
        illustrate_new_primitive grammar' (find_new_primitive grammar grammar') train_frontiers';
        (* if !verbose_compression && max_compression_steps > 1 then
          export_compression_checkpoint ~nc ~structurePenalty ~aic ~topK ~pseudoCounts ~arity ~bs ~topI g' frontiers';
        flush_everything(); *)
        compress_rewrite_report_loop (max_compression_steps - 1) grammar' train_frontiers' test_frontiers'
  in
  time_it "[ocaml] Completed ocaml compression." (fun () ->
    compress_rewrite_report_loop ~max_compression_steps grammar train_frontiers test_frontiers)
;;

