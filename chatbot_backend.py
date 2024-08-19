#import ConversationSummaryBufferMemory,ConversationChain,chatBedrock langChain modules
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_aws import ChatBedrock
#write a function for invoke model -client connection with bedrock with profile,modelId & inference params
def demo_chatbot():
    demo_llm=ChatBedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-3-haiku-20240307-v1:0',
        model_kwargs={
            "max_tokens": 300,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop_sequences": ["\n\nHuman:"]
            })
    return demo_llm

#test the llm with predict method instead of invoke method
    #return demo_llm.invoke(input_text)
#response=demo_chatbot(input_text="Hello, how are you today?")
#print(response)

#create a function for conversationSummerBuffermemory
def demo_memory():
    llm_data=demo_chatbot()
    memory=ConversationSummaryBufferMemory(
        llm=llm_data,
        max_token_limit=300
    )
    return memory
#create a conversationChain-input text+memory
def demo_conversation(input_text,memory):
    llm_chain_data=demo_chatbot()
    llm_conversation=ConversationChain(
        llm=llm_chain_data,
        memory=memory,
        verbose=True
    )
#chat response using invoke (prompt template)
    chat_reply=llm_conversation.invoke(input_text)
    return chat_reply['response']
