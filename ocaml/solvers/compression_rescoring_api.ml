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

open Compression

(** Constant strings for the JSON messages*)
let api_fn_constant = "api_fn";;
let required_args_constant = "required_args";;

let grammar_constant = "grammar";;
let frontiers_constant = "frontiers";;
let train_constant = "train";;
let test_constant = "test" ;;

(** Standard hyperparameters for the compressor. **)
let kwargs_constant = "kwargs_constant";;
let max_candidates_per_compression_step_constant = "max_candidates_per_compression_step";;
let max_compression_steps_constant = "max_compression_steps";;
let arity_constant = "arity";; 
let pseudocounts_constant = "pseudocounts";;
let aic_constant = "aic";;
let cpus_constant = "cpus";;

(** Default values for hyperparmaeters for the compressor. **)
let default_max_candidates_per_compression_step = 200;;
let default_max_compression_steps = 1000;;
let default_arity = 3;;
let default_pseudocounts = 30.0;;
let default_structure_penalty = 1.5;;

(** API Function handlers. All API functions are registered in the api_fn_handlers table. *)
let api_fn_handlers = Hashtbl.Poly.create();;
let register_api_fn api_fn_name api_fn_handler = Hashtbl.set api_fn_handlers api_fn_name api_fn_handler;;

register_api_fn "test_send_receive_response" (fun grammar train_frontiers test_frontiers kwargs ->
    let () = (Printf.eprintf "[ocaml] test_send_receive_response\n") in 
    let serialized_response = `Assoc([
      required_args_constant, `Assoc([
        grammar_constant, `List([serialize_grammar grammar]);
        frontiers_constant, `Assoc([
          train_constant, `List(train_frontiers |> List.map ~f:serialize_frontier);
          test_constant, `List(test_frontiers |> List.map ~f:serialize_frontier);
        ])
      ])
    ]) in 
    serialized_response
);;

(** TODO: make a common file and import everything **)
let deserialize_compressor_kwargs json_kwargs = 
  (** Utility function to deserialize common kwargs hyperparameters for the compressor. Sets defaults.**)
  let open Yojson.Basic.Util in
  let open Yojson.Basic in

  let max_candidates_per_compression_step = try
    json_kwargs |> member max_candidates_per_compression_step_constant |> to_int
  with _ -> default_max_candidates_per_compression_step in

  json_kwargs;;

register_api_fn "get_compressed_grammmar_and_rewritten_frontiers" (fun grammar train_frontiers test_frontiers kwargs ->
  let () = (Printf.eprintf "[ocaml] get_compressed_grammmar_and_rewritten_frontiers \n") in 
  let () = (Printf.eprintf "[ocaml] Compressing grammar and rewriting from %d train_frontiers and %d test_frontiers \n" (List.length train_frontiers) (List.length test_frontiers)) in 

  

  let serialized_response = `Assoc([
    required_args_constant, `Assoc([
      grammar_constant, `List([serialize_grammar grammar]);
      frontiers_constant, `Assoc([
        train_constant, `List(train_frontiers |> List.map ~f:serialize_frontier);
        test_constant, `List(test_frontiers |> List.map ~f:serialize_frontier);
      ])
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
  let api_fn_name = j |> member api_fn_constant |> to_string |> string_replace "\"" "" in 
  let required_args = j |> member required_args_constant in
    let grammar = required_args |> member grammar_constant |> deserialize_grammar |> strip_grammar in

    let frontiers = required_args |> member frontiers_constant in
    let train_frontiers = frontiers |> member train_constant |> to_list |> List.map ~f:deserialize_frontier in
    let test_fronters = frontiers |> member test_constant |> to_list |> List.map ~f:deserialize_frontier in
  
  let kwargs = j |> member kwargs_constant in 

  let () = (Printf.eprintf "[ocaml] deserialize_api_call_json_message: api_fn %s\n" api_fn_name) in 
  api_fn_name, grammar, train_frontiers, test_fronters, kwargs

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



  