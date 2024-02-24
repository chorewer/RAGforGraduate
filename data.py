from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.text_splitter import CharacterTextSplitter
import weaviate
from weaviate.embedded import EmbeddedOptions
vectorstore = []
def load_database(src):
    loader = TextLoader(src,encoding='utf-8')
    documents = loader.load()
    # 「加载数据」：这里选择了陶喆百度百科txt化，作为文档输入 。
    # 文档是txt文本，要加载文本这里使用 LangChain 的 TextLoader。
    
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = text_splitter.split_documents(documents)
    # 「数据分块」：因为文档在其原始状态下太长(将近5万行)，
    # 无法放入大模型的上下文窗口，所以需要将其分成更小的部分。
    # LangChain 内置了许多用于文本的分割器。这里使用 chunk_size 约为 1024 
    # 且 chunk_overlap 为128 的 CharacterTextSplitter 来保持块之间的文本连续性。
    
    

    client = weaviate.Client(
    embedded_options = EmbeddedOptions()
    )

    vectorstore = Weaviate.from_documents(
        client = client,    
        documents = chunks,
        embedding = OpenAIEmbeddings(),
        by_text = False
    )
    # 「数据块存储」：要启用跨文本块的语义搜索，
    # 需要为每个块生成向量嵌入，然后将它们与其嵌入存储在一起。
    # 要生成向量嵌入，可以使用 OpenAI 嵌入模型，
    # 并使用 Weaviate 向量数据库来进行存储。
    # 通过调用 .from_documents()，矢量数据库会自动填充块。
def question_from_documents(question):
    #「第一步：数据检索」 将数据存入矢量数据库后，
    # 就可以将其定义为检索器组件，该组件根据用户查询和嵌入块之间的语义相似性获取相关上下文。
    retriever = vectorstore.as_retriever()
    #「第二步：提示增强」 完成数据检索之后，就可以使用相关上下文来增强提示。
    # 在这个过程中需要准备一个提示模板。可以通过提示模板轻松自定义提示，如下所示。
    from langchain.prompts import ChatPromptTemplate
    template = """
    你是一个问答机器人助手，
    请使用以下检索到的上下文来回答问题，
    如果你不知道答案，就说你不知道。问题是：{question},上下文: {context},答案是:
    """
    prompt = ChatPromptTemplate.from_template(template)
    from langchain.chat_models import ChatOpenAI
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser() 
    )

    query = question
    res=rag_chain.invoke(query)
    print(f'答案：{res}')

if __name__ == '__main__':
    print("test data")
    load_database("G:/毕业设计/ragmake/tz.txt")