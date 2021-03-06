#import BaseHTTPServer
import http.server 
import json
from ocr import OCRNeuralNetwork
import numpy as np
import random
from random import randrange


#??????
HOST_NAME = 'http://127.0.0.1'
PORT_NUMBER = 8080
#??????????????????????
HIDDEN_NODE_COUNT = 15

# ?????
data_matrix = np.loadtxt(open('data.csv', 'rb'), delimiter = ',')
data_labels = np.loadtxt(open('dataLabels.csv', 'rb'))

# ???list??
data_matrix = data_matrix.tolist()
data_labels = data_labels.tolist()

# ?????5000????train_indice????????????
train_indice = list(range(5000))
# ??????
random.shuffle(train_indice)

nn = OCRNeuralNetwork(HIDDEN_NODE_COUNT, data_matrix, data_labels, train_indice);

class JSONHandler(http.server.http.requesthandler):
    """??????POST??"""
    def do_GET(self):
        response_code = 200
        response = ""
        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len);
        payload = json.loads(content);

        # ??????????????????????
        if payload.get('train'):
            nn.train(payload['trainArray'])
            nn.save()
        # ?????????????
        elif payload.get('predict'):
            try:
                print(nn.predict(data_matrix[0]))
                response = {"type":"test", "result":str(nn.predict(payload['image']))}
            except:
                response_code = 500
        else:
            response_code = 400

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if response:
            self.wfile.write(json.dump(response))
        return

if __name__ == '__main__':
    server_class = BaseHTTPServer.HTTPServer;
    httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)

    try:
        #?????
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    else:
        print("Unexpected server exception occurred.")
    finally:
        httpd.server_close()