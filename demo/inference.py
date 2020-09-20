from mxnet.image import imdecode
from gluoncv import model_zoo, data, utils
import requests
from io import BytesIO
import json
import base64

net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True, root='/tmp/')

def lambda_handler(event, context):
    try:
        print('Trying')
        print('event:', event)
        url = event['img_url']
        response = requests.get(url)
        img = imdecode(response.content)
        x, img = data.transforms.presets.yolo.transform_test([img], short=540)
        class_IDs, scores, bounding_boxs = net(x)
        print('class_IDs.shape:', class_IDs.shape)
        print('scores.shape:', scores.shape)
        print('bounding_boxs.shape:', bounding_boxs.shape)
        print('net.classes:', net.classes)
        for i in range(10):
            class_ID, score, bounding_box = int(class_IDs[0,i,0].asscalar()), scores[0,i,0].asscalar(), bounding_boxs[0,i,:].asnumpy()
            if class_ID != -1:
                print(i, net.classes[class_ID], score, bounding_box)
        output = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                            class_IDs[0], thresh=.3, class_names=net.classes)
        output.axis('off')
        f = BytesIO()
        output.figure.savefig(f, format='jpeg', bbox_inches='tight')
        return base64.b64encode(f.getvalue())
    except Exception as e:
        raise Exception(f'ProcessingError: {e}')
