from mxnet.image import imdecode
from gluoncv import model_zoo, data
import requests
import base64

net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True, root='/tmp/')

def lambda_handler(event, context):
    try:
        try:
            url = event['img_url']
            response = requests.get(url)
            img = imdecode(response.content)
            print('Read {1}x{0} image from event["img_url"]'.format(*img.shape))
        except KeyError:
            a85 = event['img_a85']
            img = imdecode(base64.a85decode(a85))
            print('Read {1}x{0} image from event["img_a85"]'.format(*img.shape))

        x, _ = data.transforms.presets.yolo.transform_test([img], short=540)
        print('Transformed to {3}x{2} image for processing'.format(*x.shape))
        untransform = [img.shape[1]/x.shape[3], img.shape[0]/x.shape[2]] * 2
        class_IDs, scores, bounding_boxs = net(x)
        
        return [
            {
                'class': net.classes[int(class_IDs[0,i,0].asscalar())],
                'score': round(float(scores[0,i,0].asscalar()), 3),
                'bounds': [
                    int(n * u) for (n, u)
                    in zip(bounding_boxs[0,i,:].asnumpy(), untransform)
                    ],
            }
            for i in range(class_IDs.shape[1])
            if scores[0,i,0].asscalar() > .3
            ]

    except Exception as e:
        raise Exception(f'ProcessingError: {e}')
