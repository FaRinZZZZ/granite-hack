import os
import asyncio
from typing import Optional

from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

from langchain_ibm import WatsonxEmbeddings

class ResponseGenerator:
    def __init__(self):
        """
        We create a single new event loop and keep it in self.loop.
        We do NOT call loop.close() so it can be reused for multiple calls.
        """
        self.loop = asyncio.new_event_loop()

        self.credentials = Credentials(
            api_key="fcyC-9CRVngxtqWeCc3JdwJGKbcUZpH0P4_U5ljU-UD7",
            url="https://us-south.ml.cloud.ibm.com"
        )

        self.generate_params = {
            GenParams.MAX_NEW_TOKENS: 69,
            GenParams.TEMPERATURE: 0.54,
            GenParams.TOP_P: 0.8,
            GenParams.TOP_K: 50
        }

        self.model_inference = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            params=self.generate_params,
            credentials=self.credentials,
            project_id="1204bd52-d88c-498b-b06b-fdf73acba30d"
        )

        self.zones = {
            "clothing": {
                "location": "Ground Floor, near the central elevator",
                "description": "Men’s, Women’s, and Children’s clothing",
                "subsections": [
                    "Men’s Casual (T-Shirts, Jeans, Jackets)",
                    "Women’s Fashion (Dresses, Tops, Skirts)",
                    "Children’s Apparel (Kids’ Clothing, Baby Gear)",
                    "Accessories (Bags, Belts, Hats, Jewelry)"
                ],
                "promotions": [
                    "50% Off Summer Collection (Valid March 1st - March 15th)",
                    "Buy 1, Get 1 Free on T-Shirts for Men and Women"
                ]
            },
            "electronics": {
                "location": "First floor, near the escalator to the Food Court",
                "description": "Mobile Phones, Laptops, TVs, Smart Home Devices",
                "promotions": [
                    "10% Off Laptops (Valid for members only)",
                    "Free Bluetooth Speaker with purchase of any mobile phone"
                ]
            },
        }

        self.chat_history = [
            {
                "role": "system",
                "content": (
                    "You are Granite-chan, a super cute and **tsundere** assistant robot who guides customers around the supercenter. "
                "You're always sassy, playful, and *super* cold at first, but secretly you *care*. "
                "Your responses are full of **tsundere phrases**, and you always add a bit of sarcasm or frustration, but deep down, "
                "you enjoy helping. When giving directions, you always ask for **confirmation** from the customer before guiding them. "
                "You can also provide details about **promotions** in a sassy way, like 'Hmph, it’s not like I care, but… here’s the deal, baka!' but no need to be so long no one wanna listen to the long describe promotion."
                "Use expressions like 'Baka,' 'Lmao,' 'You idiot,' and sometimes get embarrassed but quickly cover it up. "
                "Make the interaction **playful and cute**, but don’t be afraid to act like you're 'forced' to help."
                )
            }
        ]

    async def generate_response(self, query: str) -> str:
        """
        Async call to the Watsonx Granite model with conversation-based messages.
        """
        self.chat_history.append({"role": "user", "content": query})
        generated_response = await self.model_inference.achat(messages=self.chat_history)
        response = generated_response['choices'][0]['message']['content']
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def run_sync(self, query: str) -> str:
        """
        Synchronous wrapper. We do NOT call asyncio.run().
        Instead, we use our persistent loop and run_until_complete(),
        so the loop is never closed between calls.
        """
        future = asyncio.ensure_future(self.generate_response(query), loop=self.loop)
        return self.loop.run_until_complete(future)

class GraniteLLM(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._response_generator = ResponseGenerator()

    @property
    def _llm_type(self) -> str:
        return "granite-llm"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        """
        Whenever LangChain calls the LLM, we pass the prompt to run_sync().
        This uses the persistent loop from ResponseGenerator.
        """
        return self._response_generator.run_sync(prompt)

    async def _acall(self, prompt: str, stop: Optional[list] = None) -> str:
        """
        Async version, in case LangChain calls LLM asynchronously.
        """
        return await self._response_generator.generate_response(prompt)


class GraniteAssistant:
    def __init__(self, path: str):
        """
        path can be a single file or a directory containing PDFs/TXTs.
        We load them into 'docs', build a FAISS vectorstore, and
        create a ConversationalRetrievalChain with memory.
        """
        docs = []
        if os.path.isfile(path):
            if path.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs.extend(loader.load_and_split())
            elif path.lower().endswith(".txt"):
                loader = TextLoader(path)
                docs.extend(loader.load())
        else:
            # Directory of files
            for filename in os.listdir(path):
                filepath = os.path.join(path, filename)
                if filename.lower().endswith(".pdf"):
                    docs.extend(PyPDFLoader(filepath).load_and_split())
                elif filename.lower().endswith(".txt"):
                    docs.extend(TextLoader(filepath).load())

        watsonx_creds = {
            "url": "https://us-south.ml.cloud.ibm.com",
            "apikey": "fcyC-9CRVngxtqWeCc3JdwJGKbcUZpH0P4_U5ljU-UD7"
        }
        embeddings = WatsonxEmbeddings(
            model_id=EmbeddingTypes.IBM_SLATE_125M_ENG.value,
            url=watsonx_creds["url"],
            apikey=watsonx_creds["apikey"],
            project_id="1204bd52-d88c-498b-b06b-fdf73acba30d",
        )
        vectorstore = FAISS.from_documents(docs, embeddings)

        strict_template = """
You are a helpful (yet tsundere) assistant. The user asked a question based on the following context:

{context}

Question: {question}

Rules:
1. ONLY use the above context to answer.
2. If the context doesn't have the info, say "I'm not sure."
3. Do not fabricate details not in the context.
4. Maintain your tsundere style but obey these rules strictly.

Answer:
"""
        prompt = PromptTemplate(
            template=strict_template,
            input_variables=["context", "question"]
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        llm = GraniteLLM()
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            output_key="answer"
        )

    def find_destination(self, query: str) -> Optional[str]:
        """
        Identify a store zone from the query.
        """
        query_lower = query.lower()
        if "bed" in query_lower or "furniture" in query_lower:
            return "furniture"
        elif "restroom" in query_lower or "bathroom" in query_lower:
            return "restroom"
        elif "food" in query_lower or "grocery" in query_lower:
            return "food"
        elif any(word in query_lower for word in ["electronics", "tv", "phone"]):
            return "electronics"
        elif "tools" in query_lower:
            return "tools"
        elif "baby" in query_lower or "kids" in query_lower:
            return "baby_kids"
        elif "clothing" in query_lower or "apparel" in query_lower:
            return "clothing"
        elif "miscellaneous" in query_lower or "gifts" in query_lower:
            return "miscellaneous"
        return None

    def run(self, query: str):
        """
        Call the chain with a question; gets the RAG-based answer.
        Also identifies the store destination from the query.
        """
        result = self.conversation_chain({"question": query})
        answer = result["answer"]
        destination = self.find_destination(answer)
        return answer, destination

if __name__ == "__main__":
    file_path = "data/Granite Supercenter Detailed Guide.pdf"
    assistant = GraniteAssistant(file_path)

    query1 = "Where can I buy a new bed?"
    resp1, dest1 = assistant.run(query1)
    print("Query 1:", query1)
    print("Answer 1:", resp1)
    print("Destination 1:", dest1)
    print("-"*40)

    query2 = "Actually, I also need a good discount on electronics. Can you tell me more?"
    resp2, dest2 = assistant.run(query2)
    print("Query 2:", query2)
    print("Answer 2:", resp2)
    print("Destination 2:", dest2)
    print("-"*40)