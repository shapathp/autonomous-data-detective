from langchain_openai import ChatOpenAI

from autonomous_data_detective.core.config import get_settings


def main() -> None:
    settings = get_settings()

    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0,
        max_retries=1,
        timeout=30,
    )

    response = model.invoke(
        "Reply with exactly this text: OpenAI connection successful."
    )

    print("OpenAI API call completed.")
    print("Model:", settings.openai_model)
    print("Response:", response.content)


if __name__ == "__main__":
    main()
