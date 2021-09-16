(** compression_rescoring_api.ml | Author: Catherine Wong, Kevin Ellis.
  Exposes an API for working with program abstraction and compression functionality using an OCaml backend. This compiles to a binary (compression_rescoring_api.ml) that receives JSON messages from Python. Draws heavily on a compression implementation from compression.ml
  
  Sample usage: 
  JSON message: 
    {
      "api_fn": "get_compression_grammar_candidates_for_frontiers",
      "required_args" : {
        "initial_grammar" : serialized Grammar object; 
        "initial_frontiers" : {split : serialized Frontiers object},
      },
      "kwargs" : {other args}.
    }
  This can be piped to the binary (eg. using subprocess in Python.)
  This returns a JSON-formatted response that must be received from the binary caller (eg. again using subprocess.)
**)

open Core


(** main binary functionality: receives JSON messages from the calling binary and returns JSON messages out. **)
(* deserialize_api_call_json_message: Deserializes the JSON message from the calling Python binary. *)
let deserialize_api_call_json_message j = 
  Printf.eprintf "Deserializing API call."
  ;;

let () =
  Printf.eprintf "#### Starting the compression_rescoring_api \n ####";

  