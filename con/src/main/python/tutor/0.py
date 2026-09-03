import openai
import json

client = openai.OpenAI(
    #api_key="sk-ZY8b20V-_7k-LN2AP3vdww",
    api_key="sk-TsvfuAOjIjarJOJwoSDUOg",
    base_url="https://api.int2.net" 
)

response = client.chat.completions.create(
    #model="glm-5.2", # model to send to the proxy
    model="qwen3.8-27b-thinking",
    messages = [
        {
            "role": "user",
            #"content": "Hãy giải phương trình 2x - 3 = 7."
            #"content": "Hãy liệt kê tất cả các số nguyên tố nhỏ hơn 20."
            #"content": "Hãy in ra một ma phương bậc 3 dưới dạng bảng 3x3."
            #"content": "Liệt kê các số hoàn hảo nhỏ hơn 1000. "
            "content": "Hãy cho biết các thư viện hay dùng để lập trình web bằng Python."
        }
    ]
)

#print(response)
#print()

json_string = response.choices[0].message.content
print(json_string)