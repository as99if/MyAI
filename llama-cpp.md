## API Endpoints

### GET `/health`: Returns heath check result

**Response format**

- HTTP status code 503
  - Body: `{"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}`
  - Explanation: the model is still being loaded.
- HTTP status code 200
  - Body: `{"status": "ok" }`
  - Explanation: the model is successfully loaded and the server is ready.

### POST `/completion`: Given a `prompt`, it returns the predicted completion.

> [!IMPORTANT]
>
> This endpoint is **not** OAI-compatible. For OAI-compatible client, use `/v1/completions` instead.

*Options:*

`prompt`: Provide the prompt for this completion as a string or as an array of strings or numbers representing tokens. Internally, if `cache_prompt` is `true`, the prompt is compared to the previous completion and only the "unseen" suffix is evaluated. A `BOS` token is inserted at the start, if all of the following conditions are true:

  - The prompt is a string or an array with the first element given as a string
  - The model's `tokenizer.ggml.add_bos_token` metadata is `true`

These input shapes and data type are allowed for `prompt`:

  - Single string: `"string"`
  - Single sequence of tokens: `[12, 34, 56]`
  - Mixed tokens and strings: `[12, 34, "string", 56, 78]`

Multiple prompts are also supported. In this case, the completion result will be an array.

  - Only strings: `["string1", "string2"]`
  - Strings and sequences of tokens: `["string1", [12, 34, 56]]`
  - Mixed types: `[[12, 34, "string", 56, 78], [12, 34, 56], "string"]`

`temperature`: Adjust the randomness of the generated text. Default: `0.8`

`dynatemp_range`: Dynamic temperature range. The final temperature will be in the range of `[temperature - dynatemp_range; temperature + dynatemp_range]` Default: `0.0`, which is disabled.

`dynatemp_exponent`: Dynamic temperature exponent. Default: `1.0`

`top_k`: Limit the next token selection to the K most probable tokens.  Default: `40`

`top_p`: Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. Default: `0.95`

`min_p`: The minimum probability for a token to be considered, relative to the probability of the most likely token. Default: `0.05`

`n_predict`: Set the maximum number of tokens to predict when generating text. **Note:** May exceed the set limit slightly if the last token is a partial multibyte character. When 0, no tokens will be generated but the prompt is evaluated into the cache. Default: `-1`, where `-1` is infinity.

`n_indent`: Specify the minimum line indentation for the generated text in number of whitespace characters. Useful for code completion tasks. Default: `0`

`n_keep`: Specify the number of tokens from the prompt to retain when the context size is exceeded and tokens need to be discarded. The number excludes the BOS token.
By default, this value is set to `0`, meaning no tokens are kept. Use `-1` to retain all tokens from the prompt.

`stream`: Allows receiving each predicted token in real-time instead of waiting for the completion to finish (uses a different response format). To enable this, set to `true`.

`stop`: Specify a JSON array of stopping strings.
These words will not be included in the completion, so make sure to add them to the prompt for the next iteration. Default: `[]`

`typical_p`: Enable locally typical sampling with parameter p. Default: `1.0`, which is disabled.

`repeat_penalty`: Control the repetition of token sequences in the generated text. Default: `1.1`

`repeat_last_n`: Last n tokens to consider for penalizing repetition. Default: `64`, where `0` is disabled and `-1` is ctx-size.

`presence_penalty`: Repeat alpha presence penalty. Default: `0.0`, which is disabled.

`frequency_penalty`: Repeat alpha frequency penalty. Default: `0.0`, which is disabled.

`dry_multiplier`: Set the DRY (Don't Repeat Yourself) repetition penalty multiplier. Default: `0.0`, which is disabled.

`dry_base`: Set the DRY repetition penalty base value. Default: `1.75`

`dry_allowed_length`: Tokens that extend repetition beyond this receive exponentially increasing penalty: multiplier * base ^ (length of repeating sequence before token - allowed length). Default: `2`

`dry_penalty_last_n`: How many tokens to scan for repetitions. Default: `-1`, where `0` is disabled and `-1` is context size.

`dry_sequence_breakers`: Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted. Default: `['\n', ':', '"', '*']`

`xtc_probability`: Set the chance for token removal via XTC sampler. Default: `0.0`, which is disabled.

`xtc_threshold`: Set a minimum probability threshold for tokens to be removed via XTC sampler. Default: `0.1` (> `0.5` disables XTC)

`mirostat`: Enable Mirostat sampling, controlling perplexity during text generation. Default: `0`, where `0` is disabled, `1` is Mirostat, and `2` is Mirostat 2.0.

`mirostat_tau`: Set the Mirostat target entropy, parameter tau. Default: `5.0`

`mirostat_eta`: Set the Mirostat learning rate, parameter eta.  Default: `0.1`

`grammar`: Set grammar for grammar-based sampling.  Default: no grammar

`json_schema`: Set a JSON schema for grammar-based sampling (e.g. `{"items": {"type": "string"}, "minItems": 10, "maxItems": 100}` of a list of strings, or `{}` for any JSON). See [tests](../../tests/test-json-schema-to-grammar.cpp) for supported features.  Default: no JSON schema.

`seed`: Set the random number generator (RNG) seed.  Default: `-1`, which is a random seed.

`ignore_eos`: Ignore end of stream token and continue generating.  Default: `false`

`logit_bias`: Modify the likelihood of a token appearing in the generated text completion. For example, use `"logit_bias": [[15043,1.0]]` to increase the likelihood of the token 'Hello', or `"logit_bias": [[15043,-1.0]]` to decrease its likelihood. Setting the value to false, `"logit_bias": [[15043,false]]` ensures that the token `Hello` is never produced. The tokens can also be represented as strings, e.g. `[["Hello, World!",-0.5]]` will reduce the likelihood of all the individual tokens that represent the string `Hello, World!`, just like the `presence_penalty` does. Default: `[]`

`n_probs`: If greater than 0, the response also contains the probabilities of top N tokens for each generated token given the sampling settings. Note that for temperature < 0 the tokens are sampled greedily but token probabilities are still being calculated via a simple softmax of the logits without considering any other sampler settings. Default: `0`

`min_keep`: If greater than 0, force samplers to return N possible tokens at minimum. Default: `0`

`t_max_predict_ms`: Set a time limit in milliseconds for the prediction (a.k.a. text-generation) phase. The timeout will trigger if the generation takes more than the specified time (measured since the first token was generated) and if a new-line character has already been generated. Useful for FIM applications. Default: `0`, which is disabled.

`image_data`: An array of objects to hold base64-encoded image `data` and its `id`s to be reference in `prompt`. You can determine the place of the image in the prompt as in the following: `USER:[img-12]Describe the image in detail.\nASSISTANT:`. In this case, `[img-12]` will be replaced by the embeddings of the image with id `12` in the following `image_data` array: `{..., "image_data": [{"data": "<BASE64_STRING>", "id": 12}]}`. Use `image_data` only with multimodal models, e.g., LLaVA.

`id_slot`: Assign the completion task to an specific slot. If is -1 the task will be assigned to a Idle slot.  Default: `-1`

`cache_prompt`: Re-use KV cache from a previous request if possible. This way the common prefix does not have to be re-processed, only the suffix that differs between the requests. Because (depending on the backend) the logits are **not** guaranteed to be bit-for-bit identical for different batch sizes (prompt processing vs. token generation) enabling this option can cause nondeterministic results. Default: `true`

`return_tokens`: Return the raw generated token ids in the `tokens` field. Otherwise `tokens` remains empty. Default: `false`

`samplers`: The order the samplers should be applied in. An array of strings representing sampler type names. If a sampler is not set, it will not be used. If a sampler is specified more than once, it will be applied multiple times. Default: `["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]` - these are all the available values.

`timings_per_token`: Include prompt processing and text generation speed information in each response.  Default: `false`

`post_sampling_probs`: Returns the probabilities of top `n_probs` tokens after applying sampling chain.

`response_fields`: A list of response fields, for example: `"response_fields": ["content", "generation_settings/n_predict"]`. If the specified field is missing, it will simply be omitted from the response without triggering an error. Note that fields with a slash will be unnested; for example, `generation_settings/n_predict` will move the field `n_predict` from the `generation_settings` object to the root of the response and give it a new name.

`lora`: A list of LoRA adapters to be applied to this specific request. Each object in the list must contain `id` and `scale` fields. For example: `[{"id": 0, "scale": 0.5}, {"id": 1, "scale": 1.1}]`. If a LoRA adapter is not specified in the list, its scale will default to `0.0`. Please note that requests with different LoRA configurations will not be batched together, which may result in performance degradation.

**Response format**

- Note: In streaming mode (`stream`), only `content`, `tokens` and `stop` will be returned until end of completion. Responses are sent using the [Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) standard. Note: the browser's `EventSource` interface cannot be used due to its lack of `POST` request support.

- `completion_probabilities`: An array of token probabilities for each completion. The array's length is `n_predict`. Each item in the array has a nested array `top_logprobs`. It contains at **maximum** `n_probs` elements:
  ```json
  {
    "content": "<the generated completion text>",
    "tokens": [ generated token ids if requested ],
    ...
    "probs": [
      {
        "id": <token id>,
        "logprob": float,
        "token": "<most likely token>",
        "bytes": [int, int, ...],
        "top_logprobs": [
          {
            "id": <token id>,
            "logprob": float,
            "token": "<token text>",
            "bytes": [int, int, ...],
          },
          {
            "id": <token id>,
            "logprob": float,
            "token": "<token text>",
            "bytes": [int, int, ...],
          },
          ...
        ]
      },
      {
        "id": <token id>,
        "logprob": float,
        "token": "<most likely token>",
        "bytes": [int, int, ...],
        "top_logprobs": [
          ...
        ]
      },
      ...
    ]
  },
  ```
  Please note that if `post_sampling_probs` is set to `true`:
    - `logprob` will be replaced with `prob`, with the value between 0.0 and 1.0
    - `top_logprobs` will be replaced with `top_probs`. Each element contains:
      - `id`: token ID
      - `token`: token in string
      - `bytes`: token in bytes
      - `prob`: token probability, with the value between 0.0 and 1.0
    - Number of elements in `top_probs` may be less than `n_probs`

- `content`: Completion result as a string (excluding `stopping_word` if any). In case of streaming mode, will contain the next token as a string.
- `tokens`: Same as `content` but represented as raw token ids. Only populated if `"return_tokens": true` or `"stream": true` in the request.
- `stop`: Boolean for use with `stream` to check whether the generation has stopped (Note: This is not related to stopping words array `stop` from input options)
- `generation_settings`: The provided options above excluding `prompt` but including `n_ctx`, `model`. These options may differ from the original ones in some way (e.g. bad values filtered out, strings converted to tokens, etc.).
- `model`: The model alias (for model path, please use `/props` endpoint)
- `prompt`: The processed `prompt` (special tokens may be added)
- `stop_type`: Indicating whether the completion has stopped. Possible values are:
  - `none`: Generating (not stopped)
  - `eos`: Stopped because it encountered the EOS token
  - `limit`: Stopped because `n_predict` tokens were generated before stop words or EOS was encountered
  - `word`: Stopped due to encountering a stopping word from `stop` JSON array provided
- `stopping_word`: The stopping word encountered which stopped the generation (or "" if not stopped due to a stopping word)
- `timings`: Hash of timing information about the completion such as the number of tokens `predicted_per_second`
- `tokens_cached`: Number of tokens from the prompt which could be re-used from previous completion (`n_past`)
- `tokens_evaluated`: Number of tokens evaluated in total from the prompt
- `truncated`: Boolean indicating if the context size was exceeded during generation, i.e. the number of tokens provided in the prompt (`tokens_evaluated`) plus tokens generated (`tokens predicted`) exceeded the context size (`n_ctx`)


### POST `/tokenize`: Tokenize a given text

*Options:*

`content`: (Required) The text to tokenize.

`add_special`: (Optional) Boolean indicating if special tokens, i.e. `BOS`, should be inserted.  Default: `false`

`with_pieces`: (Optional) Boolean indicating whether to return token pieces along with IDs.  Default: `false`

**Response:**

Returns a JSON object with a `tokens` field containing the tokenization result. The `tokens` array contains either just token IDs or objects with `id` and `piece` fields, depending on the `with_pieces` parameter. The piece field is a string if the piece is valid unicode or a list of bytes otherwise.


If `with_pieces` is `false`:
```json
{
  "tokens": [123, 456, 789]
}
```

If `with_pieces` is `true`:
```json
{
  "tokens": [
    {"id": 123, "piece": "Hello"},
    {"id": 456, "piece": " world"},
    {"id": 789, "piece": "!"}
  ]
}
```

With input 'á' (utf8 hex: C3 A1) on tinyllama/stories260k
```json
{
  "tokens": [
    {"id": 198, "piece": [195]}, // hex C3
    {"id": 164, "piece": [161]} // hex A1
  ]
}
```

### POST `/detokenize`: Convert tokens to text

*Options:*

`tokens`: Set the tokens to detokenize.

### POST `/embedding`: Generate embedding of a given text

> [!IMPORTANT]
>
> This endpoint is **not** OAI-compatible. For OAI-compatible client, use `/v1/embeddings` instead.

The same as [the embedding example](../embedding) does.

*Options:*

`content`: Set the text to process.

`image_data`: An array of objects to hold base64-encoded image `data` and its `id`s to be reference in `content`. You can determine the place of the image in the content as in the following: `Image: [img-21].\nCaption: This is a picture of a house`. In this case, `[img-21]` will be replaced by the embeddings of the image with id `21` in the following `image_data` array: `{..., "image_data": [{"data": "<BASE64_STRING>", "id": 21}]}`. Use `image_data` only with multimodal models, e.g., LLaVA.

### POST `/reranking`: Rerank documents according to a given query

Similar to https://jina.ai/reranker/ but might change in the future.
Requires a reranker model (such as [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)) and the `--embedding --pooling rank` options.

*Options:*

`query`: The query against which the documents will be ranked.

`documents`: An array strings representing the documents to be ranked.

*Aliases:*
  - `/rerank`
  - `/v1/rerank`
  - `/v1/reranking`

*Examples:*

```shell
curl http://127.0.0.1:8012/v1/rerank \
    -H "Content-Type: application/json" \
    -d '{
        "model": "some-model",
            "query": "What is panda?",
            "top_n": 3,
            "documents": [
                "hi",
            "it is a bear",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
            ]
    }' | jq
```

### POST `/infill`: For code infilling.

Takes a prefix and a suffix and returns the predicted completion as stream.

*Options:*

- `input_prefix`: Set the prefix of the code to infill.
- `input_suffix`: Set the suffix of the code to infill.
- `input_extra`:  Additional context inserted before the FIM prefix.
- `prompt`:       Added after the `FIM_MID` token

`input_extra` is array of `{"filename": string, "text": string}` objects.

The endpoint also accepts all the options of `/completion`.

If the model has `FIM_REPO` and `FIM_FILE_SEP` tokens, the [repo-level pattern](https://arxiv.org/pdf/2409.12186) is used:

```txt
<FIM_REP>myproject
<FIM_SEP>{chunk 0 filename}
{chunk 0 text}
<FIM_SEP>{chunk 1 filename}
{chunk 1 text}
...
<FIM_SEP>filename
<FIM_PRE>[input_prefix]<FIM_SUF>[input_suffix]<FIM_MID>[prompt]
```

If the tokens are missing, then the extra context is simply prefixed at the start:

```txt
[input_extra]<FIM_PRE>[input_prefix]<FIM_SUF>[input_suffix]<FIM_MID>[prompt]
```

### **GET** `/props`: Get server global properties.

This endpoint is public (no API key check). By default, it is read-only. To make POST request to change global properties, you need to start server with `--props`

**Response format**

```json
{
  "default_generation_settings": {
    "id": 0,
    "id_task": -1,
    "n_ctx": 1024,
    "speculative": false,
    "is_processing": false,
    "params": {
      "n_predict": -1,
      "seed": 4294967295,
      "temperature": 0.800000011920929,
      "dynatemp_range": 0.0,
      "dynatemp_exponent": 1.0,
      "top_k": 40,
      "top_p": 0.949999988079071,
      "min_p": 0.05000000074505806,
      "xtc_probability": 0.0,
      "xtc_threshold": 0.10000000149011612,
      "typical_p": 1.0,
      "repeat_last_n": 64,
      "repeat_penalty": 1.0,
      "presence_penalty": 0.0,
      "frequency_penalty": 0.0,
      "dry_multiplier": 0.0,
      "dry_base": 1.75,
      "dry_allowed_length": 2,
      "dry_penalty_last_n": -1,
      "dry_sequence_breakers": [
        "\n",
        ":",
        "\"",
        "*"
      ],
      "mirostat": 0,
      "mirostat_tau": 5.0,
      "mirostat_eta": 0.10000000149011612,
      "stop": [],
      "max_tokens": -1,
      "n_keep": 0,
      "n_discard": 0,
      "ignore_eos": false,
      "stream": true,
      "n_probs": 0,
      "min_keep": 0,
      "grammar": "",
      "samplers": [
        "dry",
        "top_k",
        "typ_p",
        "top_p",
        "min_p",
        "xtc",
        "temperature"
      ],
      "speculative.n_max": 16,
      "speculative.n_min": 5,
      "speculative.p_min": 0.8999999761581421,
      "timings_per_token": false
    },
    "prompt": "",
    "next_token": {
      "has_next_token": true,
      "has_new_line": false,
      "n_remain": -1,
      "n_decoded": 0,
      "stopping_word": ""
    }
  },
  "total_slots": 1,
  "model_path": "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
  "chat_template": "...",
  "build_info": "b(build number)-(build commit hash)"
}
```

- `default_generation_settings` - the default generation settings for the `/completion` endpoint, which has the same fields as the `generation_settings` response object from the `/completion` endpoint.
- `total_slots` - the total number of slots for process requests (defined by `--parallel` option)
- `model_path` - the path to model file (same with `-m` argument)
- `chat_template` - the model's original Jinja2 prompt template

### POST `/props`: Change server global properties.

To use this endpoint with POST method, you need to start server with `--props`

*Options:*

- None yet

### POST `/embeddings`: non-OpenAI-compatible embeddings API

This endpoint supports all poolings, including `--pooling none`. When the pooling is `none`, the responses will contain the *unnormalized* embeddings for *all* input tokens. For all other pooling types, only the pooled embeddings are returned, normalized using Euclidian norm.

Note that the response format of this endpoint is different from `/v1/embeddings`.

*Options:*

Same as the `/v1/embeddings` endpoint.

*Examples:*

Same as the `/v1/embeddings` endpoint.

**Response format**

```json
[
  {
    "index": 0,
    "embedding": [
      [ ... embeddings for token 0   ... ],
      [ ... embeddings for token 1   ... ],
      [ ... ]
      [ ... embeddings for token N-1 ... ],
    ]
  },
  ...
  {
    "index": P,
    "embedding": [
      [ ... embeddings for token 0   ... ],
      [ ... embeddings for token 1   ... ],
      [ ... ]
      [ ... embeddings for token N-1 ... ],
    ]
  }
]
```

### GET `/slots`: Returns the current slots processing state

> [!WARNING]
> This endpoint is intended for debugging and may be modified in future versions. For security reasons, we strongly advise against enabling it in production environments.

This endpoint is disabled by default and can be enabled with `--slots`

If query param `?fail_on_no_slot=1` is set, this endpoint will respond with status code 503 if there is no available slots.

**Response format**

Example:

```json
[
  {
    "id": 0,
    "id_task": -1,
    "n_ctx": 1024,
    "speculative": false,
    "is_processing": false,
    "params": {
      "n_predict": -1,
      "seed": 4294967295,
      "temperature": 0.800000011920929,
      "dynatemp_range": 0.0,
      "dynatemp_exponent": 1.0,
      "top_k": 40,
      "top_p": 0.949999988079071,
      "min_p": 0.05000000074505806,
      "xtc_probability": 0.0,
      "xtc_threshold": 0.10000000149011612,
      "typical_p": 1.0,
      "repeat_last_n": 64,
      "repeat_penalty": 1.0,
      "presence_penalty": 0.0,
      "frequency_penalty": 0.0,
      "dry_multiplier": 0.0,
      "dry_base": 1.75,
      "dry_allowed_length": 2,
      "dry_penalty_last_n": -1,
      "dry_sequence_breakers": [
        "\n",
        ":",
        "\"",
        "*"
      ],
      "mirostat": 0,
      "mirostat_tau": 5.0,
      "mirostat_eta": 0.10000000149011612,
      "stop": [],
      "max_tokens": -1,
      "n_keep": 0,
      "n_discard": 0,
      "ignore_eos": false,
      "stream": true,
      "n_probs": 0,
      "min_keep": 0,
      "grammar": "",
      "samplers": [
        "dry",
        "top_k",
        "typ_p",
        "top_p",
        "min_p",
        "xtc",
        "temperature"
      ],
      "speculative.n_max": 16,
      "speculative.n_min": 5,
      "speculative.p_min": 0.8999999761581421,
      "timings_per_token": false
    },
    "prompt": "",
    "next_token": {
      "has_next_token": true,
      "has_new_line": false,
      "n_remain": -1,
      "n_decoded": 0,
      "stopping_word": ""
    }
  }
]
```

### GET `/metrics`: Prometheus compatible metrics exporter

This endpoint is only accessible if `--metrics` is set.

Available metrics:
- `llamacpp:prompt_tokens_total`: Number of prompt tokens processed.
- `llamacpp:tokens_predicted_total`: Number of generation tokens processed.
- `llamacpp:prompt_tokens_seconds`: Average prompt throughput in tokens/s.
- `llamacpp:predicted_tokens_seconds`: Average generation throughput in tokens/s.
- `llamacpp:kv_cache_usage_ratio`: KV-cache usage. `1` means 100 percent usage.
- `llamacpp:kv_cache_tokens`: KV-cache tokens.
- `llamacpp:requests_processing`: Number of requests processing.
- `llamacpp:requests_deferred`: Number of requests deferred.

### POST `/slots/{id_slot}?action=save`: Save the prompt cache of the specified slot to a file.

*Options:*

`filename`: Name of the file to save the slot's prompt cache. The file will be saved in the directory specified by the `--slot-save-path` server parameter.

**Response format**

```json
{
    "id_slot": 0,
    "filename": "slot_save_file.bin",
    "n_saved": 1745,
    "n_written": 14309796,
    "timings": {
        "save_ms": 49.865
    }
}
```

### POST `/slots/{id_slot}?action=restore`: Restore the prompt cache of the specified slot from a file.

*Options:*

`filename`: Name of the file to restore the slot's prompt cache from. The file should be located in the directory specified by the `--slot-save-path` server parameter.

**Response format**

```json
{
    "id_slot": 0,
    "filename": "slot_save_file.bin",
    "n_restored": 1745,
    "n_read": 14309796,
    "timings": {
        "restore_ms": 42.937
    }
}
```

### POST `/slots/{id_slot}?action=erase`: Erase the prompt cache of the specified slot.

**Response format**

```json
{
    "id_slot": 0,
    "n_erased": 1745
}
```

### GET `/lora-adapters`: Get list of all LoRA adapters

This endpoint returns the loaded LoRA adapters. You can add adapters using `--lora` when starting the server, for example: `--lora my_adapter_1.gguf --lora my_adapter_2.gguf ...`

By default, all adapters will be loaded with scale set to 1. To initialize all adapters scale to 0, add `--lora-init-without-apply`

Please note that this value will be overwritten by the `lora` field for each request.

If an adapter is disabled, the scale will be set to 0.

**Response format**

```json
[
    {
        "id": 0,
        "path": "my_adapter_1.gguf",
        "scale": 0.0
    },
    {
        "id": 1,
        "path": "my_adapter_2.gguf",
        "scale": 0.0
    }
]
```

### POST `/lora-adapters`: Set list of LoRA adapters

This sets the global scale for LoRA adapters. Please note that this value will be overwritten by the `lora` field for each request.

To disable an adapter, either remove it from the list below, or set scale to 0.

**Request format**

To know the `id` of the adapter, use GET `/lora-adapters`

```json
[
  {"id": 0, "scale": 0.2},
  {"id": 1, "scale": 0.8}
]
```

## OpenAI-compatible API Endpoints

### GET `/v1/models`: OpenAI-compatible Model Info API

Returns information about the loaded model. See [OpenAI Models API documentation](https://platform.openai.com/docs/api-reference/models).

The returned list always has one single element.

By default, model `id` field is the path to model file, specified via `-m`. You can set a custom value for model `id` field via `--alias` argument. For example, `--alias gpt-4o-mini`.

Example:

```json
{
    "object": "list",
    "data": [
        {
            "id": "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "object": "model",
            "created": 1735142223,
            "owned_by": "llamacpp",
            "meta": {
                "vocab_type": 2,
                "n_vocab": 128256,
                "n_ctx_train": 131072,
                "n_embd": 4096,
                "n_params": 8030261312,
                "size": 4912898304
            }
        }
    ]
}
```

### POST `/v1/completions`: OpenAI-compatible Completions API

Given an input `prompt`, it returns the predicted completion. Streaming mode is also supported. While no strong claims of compatibility with OpenAI API spec is being made, in our experience it suffices to support many apps.

*Options:*

See [OpenAI Completions API documentation](https://platform.openai.com/docs/api-reference/completions).

llama.cpp `/completion`-specific features such as `mirostat` are supported.

*Examples:*

Example usage with `openai` python library:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)

completion = client.completions.create(
  model="davinci-002",
  prompt="I believe the meaning of life is",
  max_tokens=8
)

print(completion.choices[0].text)
```

### POST `/v1/chat/completions`: OpenAI-compatible Chat Completions API

Given a ChatML-formatted json description in `messages`, it returns the predicted completion. Both synchronous and streaming mode are supported, so scripted and interactive applications work fine. While no strong claims of compatibility with OpenAI API spec is being made, in our experience it suffices to support many apps. Only models with a [supported chat template](https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template) can be used optimally with this endpoint. By default, the ChatML template will be used.

*Options:*

See [OpenAI Chat Completions API documentation](https://platform.openai.com/docs/api-reference/chat). While some OpenAI-specific features such as function calling aren't supported, llama.cpp `/completion`-specific features such as `mirostat` are supported.

The `response_format` parameter supports both plain JSON output (e.g. `{"type": "json_object"}`) and schema-constrained JSON (e.g. `{"type": "json_object", "schema": {"type": "string", "minLength": 10, "maxLength": 100}}` or `{"type": "json_schema", "schema": {"properties": { "name": { "title": "Name",  "type": "string" }, "date": { "title": "Date",  "type": "string" }, "participants": { "items": {"type: "string" }, "title": "Participants",  "type": "string" } } } }`), similar to other OpenAI-inspired API providers.

*Examples:*

You can use either Python `openai` library with appropriate checkpoints:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
    {"role": "user", "content": "Write a limerick about python exceptions"}
  ]
)

print(completion.choices[0].message)
```

... or raw HTTP requests:

```shell
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
"model": "gpt-3.5-turbo",
"messages": [
{
    "role": "system",
    "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
},
{
    "role": "user",
    "content": "Write a limerick about python exceptions"
}
]
}'
```

### POST `/v1/embeddings`: OpenAI-compatible embeddings API

This endpoint requires that the model uses a pooling different than type `none`. The embeddings are normalized using the Eucledian norm.

*Options:*

See [OpenAI Embeddings API documentation](https://platform.openai.com/docs/api-reference/embeddings).

*Examples:*

- input as string

  ```shell
  curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
          "input": "hello",
          "model":"GPT-4",
          "encoding_format": "float"
  }'
  ```

- `input` as string array

  ```shell
  curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
          "input": ["hello", "world"],
          "model":"GPT-4",
          "encoding_format": "float"
  }'
  ```

## More examples

### Interactive mode

Check the sample in [chat.mjs](chat.mjs).
Run with NodeJS version 16 or later:

```sh
node chat.mjs
```

Another sample in [chat.sh](chat.sh).
Requires [bash](https://www.gnu.org/software/bash/), [curl](https://curl.se) and [jq](https://jqlang.github.io/jq/).
Run with bash:

```sh
bash chat.sh
```

### OAI-like API

The HTTP `llama-server` supports an OAI-like API: https://github.com/openai/openai-openapi

### API errors

`llama-server` returns errors in the same format as OAI: https://github.com/openai/openai-openapi

Example of an error:

```json
{
    "error": {
        "code": 401,
        "message": "Invalid API Key",
        "type": "authentication_error"
    }
}
```

Apart from error types supported by OAI, we also have custom types that are specific to functionalities of llama.cpp:

**When /metrics or /slots endpoint is disabled**

```json
{
    "error": {
        "code": 501,
        "message": "This server does not support metrics endpoint.",
        "type": "not_supported_error"
    }
}
```

**When the server receives invalid grammar via */completions endpoint**

```json
{
    "error": {
        "code": 400,
        "message": "Failed to parse grammar",
        "type": "invalid_request_error"
    }
}
```