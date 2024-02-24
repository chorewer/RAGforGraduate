from bs4 import BeautifulSoup
def transfer_html_to_text(src,dst):
    with open(src, "r", encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    # print(soup.select('.lemmaWrapper_fce1P'))
    result = "[]"
    for item in soup.select('.lemmaWrapper_fce1P')[0].descendants:
        result += item.get_text()
    result.encode(encoding='utf-8')
    with open(dst, 'w',encoding='utf-8') as f:
        f.write(result)

if __name__ == '__main__':
    print("test transfer")