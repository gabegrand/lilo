(** compression_rescoring_api.ml | Author: Catherine Wong, Kevin Ellis.
  Exposes an API for working with program abstraction and compression functionality using an OCaml backend. This compiles to a binary (compression_rescoring_api.ml) that receives JSON messages from Python. Draws heavily on a compression implementation from compression.ml
  
  Sample usage: 
  JSON message: 
    {
      "api_fn": "get_compression_grammar_candidates_for_frontiers",
      "required_args" : {
        "grammar" : serialized Grammar object; 
        "frontiers" : {split : serialized Frontiers object},
      },
      "kwargs" : {other args}.
    }
  This can be piped to the binary (eg. using subprocess in Python.)
  This returns a JSON-formatted response that must be received from the binary caller (eg. again using subprocess.) in the same format.
  However:
    grammar: [array of grammars containing candidate DSLs.]
    frontiers: [array of {split : serialized Frontiers object}] containing frontiers rewritten under each corresponding DSL.

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
open Versions

open Compression_rescoring_api_utils

(** Constant strings for the JSON messages*)
let api_fn_string = "api_fn";;
let required_args_string = "required_args";;

let grammar_string = "grammar";;
let frontiers_string = "frontiers";;
let compression_scores_string = "compression_scores"
let train_string = "train";;
let test_string = "test" ;;


(** API Function handlers. All API functions are registered in the api_fn_handlers table. *)
let api_fn_handlers = Hashtbl.Poly.create();;
let register_api_fn api_fn_name api_fn_handler = Hashtbl.set api_fn_handlers api_fn_name api_fn_handler;;

(** test_send_receive_response: test handshake function for using the binary. Returns exactly the grammar and frontiers it was provided. *)
register_api_fn "test_send_receive_response" (fun grammar train_frontiers test_frontiers kwargs ->
    let () = (Printf.eprintf "[ocaml] test_send_receive_response\n") in 
    let serialized_response = `Assoc([
      required_args_string, `Assoc([
        grammar_string, `List([serialize_grammar grammar]);
        frontiers_string, `List([`Assoc(
          [
          train_string, `List(train_frontiers |> List.map ~f:serialize_frontier);
          test_string, `List(test_frontiers |> List.map ~f:serialize_frontier);
        ]
        
        )])
      ])
    ]) in 
    serialized_response
);;

(** get_compressed_grammmar_and_rewritten_frontiers: compresses the grammar with respect to the train frontiers. 
  Reference: compression implementation in compression.ml
  Returns JSON response containing: 
    grammar : [single fully compressed grammar]
    frontiers: [split: split frontiers rewritten wrt the fully compressed grammar] 
*)
register_api_fn "get_compressed_grammmar_and_rewritten_frontiers" (fun grammar train_frontiers test_frontiers kwargs ->
  let () = (Printf.eprintf "[ocaml] get_compressed_grammmar_and_rewritten_frontiers \n") in 
  let () = (Printf.eprintf "[ocaml] Compressing grammar and rewriting from %d train_frontiers and %d test_frontiers \n" (List.length train_frontiers) (List.length test_frontiers)) in 

  let max_candidates_per_compression_step, max_grammar_candidates_to_retain_for_rewriting, max_compression_steps, top_k, arity, pseudocounts, structure_penalty, aic, cpus, language_alignments_weight, language_alignments = deserialize_compressor_kwargs kwargs in

  let compressed_grammar, rewritten_train_frontiers, rewritten_test_frontiers = compress_grammar_and_rewrite_frontiers_loop ~grammar ~train_frontiers ~test_frontiers ~language_alignments ~max_candidates_per_compression_step ~max_compression_steps ~top_k ~arity ~pseudocounts ~structure_penalty ~aic ~cpus  ~language_alignments_weight in 

  let serialized_response = `Assoc([
    required_args_string, `Assoc([
      grammar_string, `List([serialize_grammar compressed_grammar]);
      frontiers_string, `List([`Assoc([
        train_string, `List(rewritten_train_frontiers |> List.map ~f:serialize_frontier);
        test_string, `List(rewritten_test_frontiers |> List.map ~f:serialize_frontier);
      ])])
    ])
  ]) in 
  serialized_response
);;

(** get_compression_grammar_candidates_and_rewritten_frontiers: compresses the grammar with respect to the train frontiers and returns the top_k grammar candidates and frontiers rewritten under each. 
  Reference: compression implementation in compression.ml
  Returns JSON response containing: 
    grammar : [up to top-k (k = max_candidates_per_compression_step) grammars containing a single library function each.]
    frontiers: [k frontier objects with
      [split: split frontiers rewritten wrt the fully compressed grammar_i]
      ] 
    scores:[k scalar compression scores corresponding to each grammar.]
*)
register_api_fn "get_compressed_grammar_candidates_and_rewritten_frontiers" (fun grammar train_frontiers test_frontiers kwargs ->
  let () = (Printf.eprintf "[ocaml] get_compressed_grammmar_candidates_and_rewritten_frontiers \n") in 
  let () = (Printf.eprintf "[ocaml] Compressing grammar and rewriting candidates from %d train_frontiers and %d test_frontiers \n" (List.length train_frontiers) (List.length test_frontiers)) in 

  let max_candidates_per_compression_step, max_grammar_candidates_to_retain_for_rewriting, max_compression_steps, top_k, arity, pseudocounts, structure_penalty, aic, cpus, language_alignments_weight, language_alignments = deserialize_compressor_kwargs kwargs in

  let compressed_grammar_candidates, compression_scores_candidates, rewritten_train_frontiers_candidates, rewritten_test_frontiers_candidates = compress_grammar_candidates_and_rewrite_frontiers_for_each ~grammar ~train_frontiers ~test_frontiers ~language_alignments ~max_candidates_per_compression_step ~max_grammar_candidates_to_retain_for_rewriting ~max_compression_steps ~top_k ~arity ~pseudocounts ~structure_penalty ~aic ~cpus  ~language_alignments_weight in 

  let serialized_frontiers = List.map2_exn rewritten_train_frontiers_candidates rewritten_test_frontiers_candidates ~f: (fun train_frontiers test_frontiers ->
    `Assoc([
        train_string, `List(train_frontiers |> List.map ~f:serialize_frontier);
        test_string, `List(test_frontiers |> List.map ~f:serialize_frontier);
      ])
    ) in 
  let serialized_response = `Assoc([
    required_args_string, `Assoc([
      grammar_string, `List(compressed_grammar_candidates |> List.map ~f:serialize_grammar);
      frontiers_string, `List(serialized_frontiers);
      compression_scores_string, `List(compression_scores_candidates |> List.map ~f:(fun score -> `Float(score)))
    ])
  ]) in 
  serialized_response
);;


(** Main binary functionality: receives JSON messages from the calling binary and returns JSON messages out. **)

let string_replace input output =
  Str.global_replace (Str.regexp_string input) output;;

let deserialize_api_call_json_message j = 
  (* deserialize_api_call_json_message: Deserializes the JSON message from the calling Python binary.
  :ret: api_fn: string name of API fn. 
        grammar: deserialized grammar.
        train_frontiers, test_frontiers: list of deserialized Frontiers.
        kwargs : JSON serialized kwargs. These must be serialized by the handler.
  *)
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  (** TODO: why does this come out in quotes?! *)
  let api_fn_name = j |> member api_fn_string |> to_string |> string_replace "\"" "" in 
  let required_args = j |> member required_args_string in
    let grammar = required_args |> member grammar_string |> deserialize_grammar |> strip_grammar in

    let frontiers = required_args |> member frontiers_string in
    let train_frontiers = frontiers |> member train_string |> to_list |> List.map ~f:deserialize_frontier in
    let test_frontiers = frontiers |> member test_string |> to_list |> List.map ~f:deserialize_frontier in
  
  let kwargs = j |> member kwargs_string in 

  let () = (Printf.eprintf "[ocaml] deserialize_api_call_json_message: api_fn %s\n" api_fn_name) in 
  api_fn_name, grammar, train_frontiers, test_frontiers, kwargs

let () =
  let () = Printf.eprintf "####[ocaml] called: compression_rescoring_api####\n " in 

  (* Deserialize the API function call.*)
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let json_serialized_binary_message =
    if Array.length Sys.argv > 1 then
      (assert (Array.length Sys.argv = 2);
       Yojson.Basic.from_file Sys.argv.(1))
    else 
      Yojson.Basic.from_channel Pervasives.stdin
  in 
  let api_fn_name, grammar, train_frontiers, test_frontiers, kwargs = deserialize_api_call_json_message json_serialized_binary_message in 

  (* Dispatch the appropriate function caller. *)
  let api_fn_result = 
  (try
    match api_fn_name |> Hashtbl.find api_fn_handlers with 
    | Some(api_fn_handler) -> api_fn_handler grammar train_frontiers test_frontiers kwargs
    | None -> (Printf.eprintf " [ocaml] Error: Could not find api_fn_handler for %s\n" api_fn_name;
    exit 1)
  with _ -> exit 1)
  in 
  pretty_to_string api_fn_result |> print_string



  