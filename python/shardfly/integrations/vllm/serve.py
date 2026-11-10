# Copyright 2026 ShardFly contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions derived from the Awex project (https://github.com/inclusionAI/Awex),
# Copyright the Awex developers, licensed under the Apache License, Version 2.0.

"""vLLM OpenAI API server entrypoint with ShardFly routes registered."""

import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

try:
    from vllm.entrypoints.utils import cli_env_setup
except ImportError:
    # Newer vLLM releases perform this setup in their CLI entrypoint.
    def cli_env_setup():
        pass


try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

from shardfly import logging
from shardfly.integrations.vllm.plugin import register_shardfly_routes

logger = logging.getLogger(__name__)


def main():
    # Register ShardFly routes + worker patches directly (not via vLLM plugin system).
    register_shardfly_routes()

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server (ShardFly)."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    if not args.worker_extension_cls:
        args.worker_extension_cls = (
            "shardfly.integrations.vllm.worker_extension.ShardFlyWorkerExtension"
        )
    validate_parsed_serve_args(args)

    logger.info("Starting vLLM server with ShardFly routes enabled")
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
