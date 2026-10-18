import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import main


class LocalConfigurationTests(unittest.TestCase):
    def test_uses_only_local_model(self):
        with (
            patch.dict(os.environ, {"FOUNDRY_LOCAL_MODEL": " phi-4-mini "}, clear=True),
            patch.object(main, "create_local_client") as client,
            patch.object(main, "Agent") as agent,
            patch.object(main, "ResponsesHostServer") as server,
            patch.object(main, "MCPStreamableHTTPTool") as mcp,
        ):
            agent.return_value.__aenter__.return_value = agent.return_value
            server.return_value.run_async = AsyncMock()
            main.main()

            client.assert_called_once_with(model="phi-4-mini")
            self.assertIs(agent.call_args.kwargs["client"], client.return_value)
            self.assertEqual(agent.call_args.kwargs["default_options"], {"store": False})
            self.assertEqual(
                agent.call_args.kwargs["tools"],
                [*main.HERRAMIENTAS_INVENTARIO, mcp.return_value],
            )
            self.assertEqual(agent.call_args.kwargs["instructions"], main.INSTRUCTIONS)
            self.assertEqual(agent.call_args.kwargs["name"], "RosterStock")
            mcp.assert_called_once_with(
                name="Microsoft Learn",
                url="https://learn.microsoft.com/api/mcp",
                allowed_tools=[
                    "microsoft_docs_search",
                    "microsoft_docs_fetch",
                    "microsoft_code_sample_search",
                ],
                load_prompts=False,
                request_timeout=60,
                approval_mode="never_require",
            )
            server.assert_called_once_with(agent.return_value)
            server.return_value.run_async.assert_awaited_once_with()
            agent.return_value.__aexit__.assert_awaited_once_with(None, None, None)

    def test_missing_or_blank_local_model_fails_even_with_azure_configuration(self):
        for value in (None, "", "   "):
            environment = {
                "AZURE_AI_MODEL_DEPLOYMENT_NAME": "cloud-model",
                "FOUNDRY_MODEL_NAME": "cloud-model",
                "FOUNDRY_PROJECT_ENDPOINT": "https://example.invalid",
            }
            if value is not None:
                environment["FOUNDRY_LOCAL_MODEL"] = value
            with (
                self.subTest(value=value),
                patch.dict(os.environ, environment, clear=True),
                patch.object(main, "create_local_client") as client,
            ):
                with self.assertRaisesRegex(RuntimeError, "FOUNDRY_LOCAL_MODEL"):
                    main.main()
                client.assert_not_called()

    def test_local_runtime_failure_is_not_hidden(self):
        with (
            patch.dict(os.environ, {"FOUNDRY_LOCAL_MODEL": "phi-4-mini"}, clear=True),
            patch.object(main, "create_local_client", side_effect=ValueError("Unavailable model")),
            patch.object(main, "ResponsesHostServer") as server,
        ):
            with self.assertRaisesRegex(ValueError, "Unavailable model"):
                main.main()
            server.assert_not_called()

    def test_discovers_local_endpoint_and_explicitly_configures_openai_client(self):
        with (
            patch.object(main.shutil, "which", return_value="foundry.exe"),
            patch.object(main.subprocess, "run") as run,
            patch.object(
                main.subprocess,
                "check_output",
                return_value='{"running":true,"webUrls":["http://127.0.0.1:61640"]}',
            ),
            patch.object(main, "AsyncOpenAI") as openai,
            patch.object(main, "OpenAIChatCompletionClient") as client,
            patch.object(
                main, "resolve_model_id", return_value="Phi-4-mini-instruct-generic-gpu:5"
            ) as resolve,
        ):
            main.create_local_client("phi-4-mini")
            self.assertEqual(run.call_count, 2)
            run.assert_any_call(["foundry.exe", "model", "load", "phi-4-mini"], check=True)
            resolve.assert_called_once_with("http://127.0.0.1:61640", "phi-4-mini")
            openai.assert_called_once_with(
                base_url="http://127.0.0.1:61640/v1", api_key="foundry-local"
            )
            client.assert_called_once_with(
                model="Phi-4-mini-instruct-generic-gpu:5", async_client=openai.return_value
            )

    def test_rejects_remote_endpoint(self):
        with (
            patch.object(main.shutil, "which", return_value="foundry.exe"),
            patch.object(main.subprocess, "run"),
            patch.object(
                main.subprocess,
                "check_output",
                return_value='{"running":true,"webUrls":["https://api.openai.com"]}',
            ),
            patch.object(main, "AsyncOpenAI") as openai,
        ):
            with self.assertRaisesRegex(RuntimeError, "loopback"):
                main.create_local_client("phi-4-mini")
            openai.assert_not_called()


class ModelResolutionTests(unittest.TestCase):
    def _with_models(self, *ids):
        models = MagicMock()
        models.models.list.return_value.data = [MagicMock(id=i) for i in ids]
        return patch.object(main, "OpenAI", return_value=models)

    def test_alias_resolves_to_instruct_model_not_reasoning(self):
        with self._with_models(
            "Phi-4-mini-reasoning-generic-gpu:3", "Phi-4-mini-instruct-generic-gpu:5"
        ):
            self.assertEqual(
                main.resolve_model_id("http://127.0.0.1:1", "phi-4-mini"),
                "Phi-4-mini-instruct-generic-gpu:5",
            )

    def test_exact_id_is_kept(self):
        with self._with_models("Phi-4-mini-instruct-generic-cpu:5"):
            self.assertEqual(
                main.resolve_model_id("http://127.0.0.1:1", "phi-4-mini-instruct-generic-cpu:5"),
                "Phi-4-mini-instruct-generic-cpu:5",
            )

    def test_unknown_or_unreachable_returns_configured_value(self):
        with self._with_models("gemma-4-e2b-it-generic-gpu:3"):
            self.assertEqual(main.resolve_model_id("http://127.0.0.1:1", "phi-4-mini"), "phi-4-mini")
        with patch.object(main, "OpenAI", side_effect=ConnectionError("down")):
            self.assertEqual(main.resolve_model_id("http://127.0.0.1:1", "phi-4-mini"), "phi-4-mini")


class MCPLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_can_be_disabled(self):
        with (
            patch.dict(os.environ, {"ENABLE_MICROSOFT_LEARN_MCP": "false"}),
            patch.object(main, "MCPStreamableHTTPTool") as mcp,
            patch.object(main, "Agent") as agent,
            patch.object(main, "ResponsesHostServer") as server,
        ):
            agent.return_value.__aenter__.return_value = agent.return_value
            server.return_value.run_async = AsyncMock()
            await main.serve_agent(None)
            mcp.assert_not_called()
            self.assertEqual(agent.call_args.kwargs["tools"], main.HERRAMIENTAS_INVENTARIO)

    async def test_closes_agent_context_when_server_fails(self):
        with (
            patch.object(main, "MCPStreamableHTTPTool"),
            patch.object(main, "Agent") as agent,
            patch.object(main, "ResponsesHostServer") as server,
        ):
            agent.return_value.__aenter__.return_value = agent.return_value
            failure = RuntimeError("Server failed")
            server.return_value.run_async = AsyncMock(side_effect=failure)

            with self.assertRaisesRegex(RuntimeError, "Server failed"):
                await main.serve_agent(None)

            agent.return_value.__aexit__.assert_awaited_once()
            self.assertIs(agent.return_value.__aexit__.call_args.args[1], failure)

    async def test_mcp_connection_failure_prevents_server_startup(self):
        with (
            patch.object(main, "MCPStreamableHTTPTool"),
            patch.object(main, "Agent") as agent,
            patch.object(main, "ResponsesHostServer") as server,
        ):
            agent.return_value.__aenter__.side_effect = RuntimeError("MCP unavailable")
            with self.assertRaisesRegex(RuntimeError, "MCP unavailable"):
                await main.serve_agent(None)
            server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
