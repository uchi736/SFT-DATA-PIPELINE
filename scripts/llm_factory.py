"""
LLMプロバイダー切り替えファクトリ

LLM_PROVIDER 環境変数で Azure OpenAI / vLLM を切り替える
- azure (デフォルト): Azure OpenAI API
- vllm: vLLM の OpenAI 互換 API

cpt-data-pipeline/scripts/llm_factory.py からのコピー（RAFT パイプライン用）。
"""

import os
import logging

logger = logging.getLogger(__name__)


def get_request_timeout() -> float:
    """LLMリクエストのタイムアウト秒数（VLLM_TIMEOUT、既定300）"""
    return float(os.getenv("VLLM_TIMEOUT", "300"))


def get_provider() -> str:
    """現在のLLMプロバイダーを返す"""
    return os.getenv("LLM_PROVIDER", "azure").lower()


def get_model_name() -> str:
    """LLM_PROVIDERに基づきモデル名を返す"""
    if get_provider() == "vllm":
        return os.getenv("VLLM_MODEL", "openai/gpt-oss-20b")
    return os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")


def create_openai_client(async_mode: bool = False):
    """
    LLM_PROVIDERに基づきOpenAIクライアントを生成

    Args:
        async_mode: Trueの場合は非同期クライアントを返す

    Returns:
        OpenAI or AsyncOpenAI (vllm) / AzureOpenAI or AsyncAzureOpenAI (azure)
    """
    provider = get_provider()

    if provider == "vllm":
        from openai import OpenAI, AsyncOpenAI
        Client = AsyncOpenAI if async_mode else OpenAI
        client = Client(
            base_url=os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            timeout=get_request_timeout(),
            max_retries=0,
        )
        logger.info(f"LLM client created: vLLM ({client.base_url})")
        return client
    else:
        from openai import AzureOpenAI, AsyncAzureOpenAI
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        if not endpoint or not api_key:
            logger.warning("Azure OpenAI credentials not found")
            return None
        Client = AsyncAzureOpenAI if async_mode else AzureOpenAI
        client = Client(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=get_request_timeout(),
            max_retries=0,
        )
        logger.info(f"LLM client created: Azure OpenAI ({endpoint})")
        return client


def create_pydanticai_agent(output_type, system_prompt: str, **kwargs):
    """
    LLM_PROVIDERに基づきPydanticAI Agentを生成

    Args:
        output_type: Pydantic出力スキーマクラス
        system_prompt: システムプロンプト
        **kwargs: Agent追加引数

    Returns:
        pydantic_ai.Agent
    """
    from pydantic_ai import Agent

    provider = get_provider()

    if provider == "vllm":
        import httpx
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.models.openai import OpenAIChatModel
        p = OpenAIProvider(
            base_url=os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            http_client=httpx.AsyncClient(timeout=get_request_timeout()),
        )
        model = OpenAIChatModel(get_model_name(), provider=p)
        logger.info(f"PydanticAI Agent created: vLLM ({get_model_name()})")
    else:
        # PydanticAI の azure: プレフィクスが参照する環境変数を設定
        os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT', '')
        os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY', '')
        os.environ['OPENAI_API_VERSION'] = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
        model = f"azure:{get_model_name()}"
        logger.info(f"PydanticAI Agent created: Azure ({get_model_name()})")

    return Agent(model, output_type=output_type, system_prompt=system_prompt, **kwargs)
