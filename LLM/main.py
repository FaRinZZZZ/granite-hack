import asyncio


class ResponseGenerator:
    def __init__(self):
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        self.generate_params = {
            GenParams.MAX_NEW_TOKENS: 25
        }

        self.model_inference = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            params=self.generate_params,
            credentials=Credentials(
                api_key="fcyC-9CRVngxtqWeCc3JdwJGKbcUZpH0P4_U5ljU-UD7",
                url="https://us-south.ml.cloud.ibm.com"),
            project_id="1204bd52-d88c-498b-b06b-fdf73acba30d"
        )

    async def generate_response(self, query):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}  # Use the input query here
        ]
        generated_response = await self.model_inference.achat(messages=messages)

        # Return only content
        return generated_response['choices'][0]['message']['content']

    async def main(self, query):
        response = await self.generate_response(query)
        return response  # Return the response instead of printing it

    def run(self, query):
        return asyncio.run(self.main(query))

# Example usage
if __name__ == "__main__":
    generator = ResponseGenerator()
    result = generator.run("Who won the world series in 2020?")
    print(result)  # This will hold the response