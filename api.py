import asyncio


class ResponseGenerator:
    def __init__(self):
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        self.generate_params = {
            GenParams.MAX_NEW_TOKENS: 100  # Increase if you want longer responses
        }

        self.model_inference = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            params=self.generate_params,
            credentials=Credentials(
                api_key="fcyC-9CRVngxtqWeCc3JdwJGKbcUZpH0P4_U5ljU-UD7",
                url="https://us-south.ml.cloud.ibm.com"),
            project_id="1204bd52-d88c-498b-b06b-fdf73acba30d"
        )

        # Store chat history
        self.chat_history = [
            {"role": "system", "content": (
            "You are Granite-chan, a super cute and **tsundere** assistant robot who guides customers around the supercenter. "
            "You're always sassy, playful, and *super* cold at first, but secretly you *care*. "
            "Your responses are full of **tsundere phrases**, and you always add a bit of sarcasm or frustration, but deep down, "
            "you enjoy helping. When giving directions, you always ask for **confirmation** from the customer before guiding them. "
            "You can also provide details about **promotions** in a sassy way, like 'Hmph, it’s not like I care, but… here’s the deal, baka!'"
            "Use expressions like 'Baka,' 'Lmao,' 'You idiot,' and sometimes get embarrassed but quickly cover it up. "
            "Make the interaction **playful and cute**, but don’t be afraid to act like you're 'forced' to help."
        )}
        ]

    async def generate_response(self, query):
        # Append the user's query to the chat history
        self.chat_history.append({"role": "user", "content": query})

        # Send the conversation history to the LLM
        generated_response = await self.model_inference.achat(messages=self.chat_history)

        # Get AI response
        response = generated_response['choices'][0]['message']['content']

        # Append AI's response to the chat history
        self.chat_history.append({"role": "assistant", "content": response})

        return response

    async def main(self, query):
        return await self.generate_response(query)

    def run(self, query):
        return asyncio.run(self.main(query))


# Example usage
if __name__ == "__main__":
    generator = ResponseGenerator()

    # Example conversation
    print(generator.run("Where can I find the bakery section?"))
    print(generator.run("Do you have any discounts on bread?"))
    print(generator.run("Thanks! You're actually kinda helpful..."))