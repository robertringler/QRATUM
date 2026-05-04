# ChatGPT Export Schema Notes

The official ChatGPT export currently ships as a `conversations.json` file containing a list
of conversation dictionaries. Each conversation describes the UI tree via the `mapping`
field, where each entry contains a `message` payload. Individual message payloads typically
look like this:

```json
{
  "id": "msg-123",
  "author": {"role": "user"},
  "create_time": 1_687_000_000.123,
  "metadata": {"model_slug": "gpt-4"},
  "content": {
    "content_type": "text",
    "parts": ["Hello world"]
  }
}
```

Earlier exports also contained a top-level `messages` array instead of a mapping. The
loader/normalizer in this package handles both shapes and falls back to a best-effort parse
if new export versions introduce additional fields.
