import dotenv,os
from transfer import transfer_html_to_text
import data
dotenv.load_dotenv()
OPENAI_API_PATH = os.getenv('OPENAI_API_PATH')
input_file = input("源文件名(*.html):")
output_file = input("目的文件名(*.txt):")
transfer_html_to_text(input_file,output_file)
data.load_database(output_file)
question = input("Question is :")
data.question_from_documents(question)



if __name__ == '__main__':
    print("test main")