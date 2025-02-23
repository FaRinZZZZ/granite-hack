import asyncio


class ResponseGenerator:
    def __init__(self):
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        # Set up the model for response generation
        self.generate_params = {
            GenParams.MAX_NEW_TOKENS: 50  # Shorten response length
        }

        self.model_inference = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            params=self.generate_params,
            credentials=Credentials(
                api_key="fcyC-9CRVngxtqWeCc3JdwJGKbcUZpH0P4_U5ljU-UD7",
                url="https://us-south.ml.cloud.ibm.com"),
            project_id="1204bd52-d88c-498b-b06b-fdf73acba30d"
        )

        # Define supercenter zones and their details (shortened)
        self.zones = {
            "clothing": {
                "location": "Ground Floor, near the elevator",
                "description": "Men’s, Women’s, and Children’s clothing",
                "promotions": [
                    "50% Off Summer Collection",
                    "Buy 1, Get 1 Free on T-Shirts"
                ]
            },
            "electronics": {
                "location": "First floor, near the escalator",
                "description": "Mobile Phones, Laptops, TVs",
                "promotions": [
                    "10% Off Laptops",
                    "Free Bluetooth Speaker with purchase of mobile phone"
                ]
            },
            "food": {
                "location": "Ground floor, near home essentials",
                "description": "Produce, Snacks, Dairy",
                "promotions": [
                    "10% Off Fresh Produce",
                    "Buy 2, Get 1 Free on Snacks"
                ]
            },
            "restroom": {
                "location": "Ground Floor, near customer service",
                "description": "Restrooms for both men and women",
                "promotions": []
            },
            "furniture": {
                "location": "Second floor, past the Electronics Section",
                "description": "Living Room & Bedroom Furniture",
                "promotions": [
                    "20% Off All Furniture",
                    "Buy 2 Rugs, Get 1 Free"
                ]
            },
            "tools": {
                "location": "Lower Ground Floor, across from Snacks",
                "description": "Hand Tools, Power Tools, Gardening",
                "promotions": [
                    "30% Off All Tools",
                    "Free Tool Bag with any power tool purchase"
                ]
            },
            "baby_kids": {
                "location": "Ground Floor, near checkout",
                "description": "Baby Clothes, Gear, Toys",
                "promotions": [
                    "10% Off All Baby Gear",
                    "Buy 1, Get 1 Free on baby clothes"
                ]
            },
            "miscellaneous": {
                "location": "Ground floor, near customer service",
                "description": "Seasonal Items, Gifts, Clearance",
                "promotions": [
                    "Up to 70% Off Clearance",
                    "Exclusive VIP Discounts"
                ]
            }
        }

        # Store chat history
        self.chat_history = [
            {"role": "system", "content": (
                "You are Granite-chan, a super cute and **tsundere** assistant robot who guides customers around the supercenter. "
                "You're always sassy, playful, and *super* cold at first, but secretly you *care*. "
                "Your responses are full of **tsundere phrases**, and you always add a bit of sarcasm or frustration, but deep down, "
                "you enjoy helping. When giving directions, you always ask for **confirmation** from the customer before guiding them. "
                "You can also provide details about **promotions** in a sassy way, like 'Hmph, it’s not like I care, but… here’s the deal, baka!' but no need to be so long no one wanna listen to the long describe promotion."
                "Use expressions like 'Baka,' 'Lmao,' 'You idiot,' and sometimes get embarrassed but quickly cover it up. "
                "Make the interaction **playful and cute**, but don’t be afraid to act like you're 'forced' to help."
            )}
        ]

    async def generate_response(self, query):
        # Find the destination based on the user's query
        destination = self.find_destination(query)

        # Append the user's query to the chat history
        self.chat_history.append({"role": "user", "content": query})

        # Send the conversation history to the LLM
        generated_response = await self.model_inference.achat(messages=self.chat_history)

        # Get AI response (limit to shorter responses)
        response = generated_response['choices'][0]['message']['content']

        # Append AI's response to the chat history
        self.chat_history.append({"role": "assistant", "content": response})

        return response, destination

    def find_destination(self, query):
        """Determine the destination based on the user's input."""
        query = query.lower()
        
        # Check for matching keywords in the query
        if "bed" in query or "furniture" in query:
            return "furniture"
        elif "restroom" in query or "bathroom" in query:
            return "restroom"
        elif "food" in query or "grocery" in query:
            return "food"
        elif "electronics" in query or "tv" in query or "phone" in query:
            return "electronics"
        elif "tools" in query:
            return "tools"
        elif "baby" in query or "kids" in query:
            return "baby_kids"
        elif "clothing" in query or "apparel" in query:
            return "clothing"
        elif "miscellaneous" in query or "gifts" in query:
            return "miscellaneous"
        else:
            return None

    async def main(self, query):
        response, destination = await self.generate_response(query)
        return response, destination

    def run(self, query):
        response, destination = asyncio.run(self.main(query))
        return response, destination


# Example usage
if __name__ == "__main__":
    generator = ResponseGenerator()

    # Example conversation
    response, destination = generator.run("I would like to go buy my new bed")
    print(f"User: I would like to go buy my new bed")
    print(f"Granite-chan: {response}")
    print(f"Destination: {destination}")

    response, destination = generator.run("Where is the restroom?")
    print(f"User: Where is the restroom?")
    print(f"Granite-chan: {response}")
    print(f"Destination: {destination}")
