# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import MutableSequence
import copy
import dataclasses
import json
import re
import logging

import google.generativeai as genai
from google.generativeai import types as genai_types

# --- Logger Setup ---
logger = logging.getLogger("controller")

@dataclasses.dataclass
class ContentChunk:
  content: str
  role: str
  tags: list[str] = dataclasses.field(default_factory=list)
  hidden: bool = False

  def format(self):
    return f"<start_of_turn>{self.role}\n{self.content}<end_of_turn>\n"


class Transcript(MutableSequence):
  def __init__(self, log_filename: str | None = None):
    self._log_filename = log_filename
    self._list = list()

  def _log_to_file(self, v):
    if isinstance(v, ContentChunk) and self._log_filename:
      json_str = dataclasses.asdict(v)
      with open(self._log_filename, 'a') as f:
        f.write(json.dumps(json_str) + "\n")

  def hide_by_tag(self, tags: list[str]):
    """
    Hides all chunks whose tags match any of the specified tags. Hidden chunks
    are not included in the string representation of the transcript.
    """
    for chunk in self._list:
      if any(tag in chunk.tags for tag in tags):
        chunk.hidden = True

  def unhide_all_tags(self):
    """Unhides all chunks in the transcript."""
    for chunk in self._list:
      chunk.hidden = False

  def format(self):
    output_text = []
    for chunk in self._list:
      if chunk.hidden:
        continue
      output_text.append(chunk.format())
    return "".join(output_text)

  def log_debug_message(self, message: str):
    c = ContentChunk(
      content=message,
      role="info",
      tags=["debug_message"]
    )
    self._log_to_file(c)

  def __len__(self): return len(self._list)
  def __getitem__(self, i): return self._list[i]
  def __delitem__(self, i): del self._list[i]
  def __str__(self): return str(self._list)
  def __setitem__(self, i, v):
    self._list[i] = v

  def insert(self, i, v):
    if i == len(self._list):
      # We have not logged this item yet, so log it now.
      self._log_to_file(v)
    self._list.insert(i, v)


def generate_completion(
  llm_name: str,
  transcript: Transcript,
  config: dict,
):
  model = genai.GenerativeModel(llm_name)
  prompt = transcript.format()

  llm_config = config['llm']

  token_count = model.count_tokens(prompt).total_tokens
  logger.info(f"generate_completion: Raw prompt has {token_count} tokens.")

  max_length = llm_config['context_size'] - llm_config['safety_margin']
  final_prompt = prompt

  if token_count > max_length:
    logger.warning(
      f"generate_completion: Truncating prompt from {token_count} to "
      f"~{max_length} tokens by removing from the beginning."
    )
    # Rebuild the prompt from chunks
    truncated_chunks = []
    current_tokens = 0
    for chunk in reversed(transcript):
      if chunk.hidden:
        continue
      chunk_text = chunk.format()

      chunk_token_count = model.count_tokens(chunk_text).total_tokens
      if current_tokens + chunk_token_count > max_length:
        break
      truncated_chunks.insert(0, chunk)  # Prepend to maintain order
      current_tokens += chunk_token_count

    final_prompt = "".join(c.format() for c in truncated_chunks)
    final_prompt = (
      f"[SYSTEM: Start of conversation was truncated due to context limit]\n"
      f"{final_prompt}"
    )

  final_prompt += "<start_of_turn>model\n"

  logger.debug("generate_completion: Final prompt to LLM:")
  final_tokens_sent = model.count_tokens(final_prompt).total_tokens
  logger.debug(f"\n{final_prompt}")
  logger.debug(
    f"generate_completion: FINAL LENGTH = {final_tokens_sent} TOKENS"
  )

  for i in range(llm_config['max_try_count']):
    try:
      response = model.generate_content(
          final_prompt,
          generation_config=genai_types.GenerationConfig(top_p=0.95, top_k=64)
      )
      if not response.candidates:
          logger.error(f"Gemini API returned no candidates. Prompt feedback: {response.prompt_feedback}")
          return
      # logger.debug(f"responses are {response}")
      output_text = response.text
      break
    except Exception as e:
      logger.error(f"Gemini API call failed in iter {i}: {e}")
      logger.error(f"Failed prompt: {final_prompt}")
      return

  logger.debug("generate_completion: LLM Response:")
  logger.debug(f"\n{output_text}")
  response_tokens = model.count_tokens(output_text).total_tokens
  logger.debug(f"generate_completion: RESPONSE = {response_tokens} TOKENS")

  output_text = output_text.replace("<end_of_turn>", "")
  output_text = output_text.replace("<start_of_turn>", "")
  output_text = output_text.strip()
  return output_text


def extract_code_blocks(markdown_string: str) -> list[str]:
  # markdown_string = markdown_string.replace("```c++", "```cpp")
  if not markdown_string:
    return None
  code_blocks = re.findall(
    r'```(?:[a-zA-Z0-9_+\.-]+)?\n(.*?)\n```', markdown_string, re.DOTALL
  )
  return [block for block in code_blocks]
