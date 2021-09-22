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
  This returns a JSON-formatted response that must be received from the binary caller (eg. again using subprocess.)

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
(** API Function handlers. All API functions are registered in the api_fn_handlers table. *)
let api_fn_handlers = Hashtbl.Poly.create();;
let register_api_fn api_fn_name api_fn_handler = Hashtbl.set api_fn_handlers api_fn_name api_fn_handler;;

register_api_fn "test_send_receive_response" (fun grammar ->
    let () = (Printf.eprintf "[ocaml] test_send_receive_response\n") in 
    let serialized_response = `Assoc(["grammar",serialize_grammar grammar]) in 
    serialized_response
);;



(** Main binary functionality: receives JSON messages from the calling binary and returns JSON messages out. **)

let string_replace input output =
  Str.global_replace (Str.regexp_string input) output;;

let deserialize_api_call_json_message j = 
  (* deserialize_api_call_json_message: Deserializes the JSON message from the calling Python binary.
  :ret: api_fn: string name of API fn. 
        grammar: deserialized grammar.
        frontiers: deserialized frontiers.
        kwargs : JSON serialized kwargs. These must be serialized by the handler.
  *)
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  (** TODO: why does this come out in quotes?! *)
  let api_fn_name = j |> member "api_fn" |> to_string |> string_replace "\"" "" in 
  let required_args = j |> member "required_args" in
    let grammar = required_args |> member "grammar" |> deserialize_grammar |> strip_grammar in
  
  let () = (Printf.eprintf "[ocaml] deserialize_api_call_json_message: api_fn %s\n" api_fn_name) in 
  api_fn_name, grammar

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
  let api_fn_name, grammar = deserialize_api_call_json_message json_serialized_binary_message in 

  (* Dispatch the appropriate function caller. *)
  let api_fn_result = 
  (try
    match api_fn_name |> Hashtbl.find api_fn_handlers with 
    | Some(api_fn_handler) -> api_fn_handler grammar
    | None -> (Printf.eprintf " [ocaml] Error: Could not find api_fn_handler for %s\n" api_fn_name;
    exit 1)
  with _ -> exit 1)
  in 
  pretty_to_string api_fn_result |> print_string



  