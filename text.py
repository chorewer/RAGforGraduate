from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
loader = TextLoader("./tz.txt")
try:
    documents = loader.load()
except RuntimeError as e:
    print(f"Error loading documents: {e}")
    # 进一步打印或记录更多详细信息，以便定位问题
    raise  # 继续抛出异常以中断程序执行并显示完整的错误追溯信息
