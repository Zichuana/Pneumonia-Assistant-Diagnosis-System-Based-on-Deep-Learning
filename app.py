import os
import io
import json
import hashlib
import cv2
import uuid
import datetime
from dataclasses import dataclass

from matplotlib import pyplot as plt
import nibabel
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, session, g, redirect, url_for
from flask_cors import CORS
from model import resnet34
from models import User, FadeBack, UserLog
from extension import db
import tensorflow as tf
import numpy as np
from flask_login import UserMixin, LoginManager, login_required, logout_user, login_user, current_user

from keras.utils import load_img
import keras.models
from keras.utils import image_utils
from werkzeug.security import check_password_hash, generate_password_hash

from datetime import timedelta
import SimpleITK as sitk
import nibabel as nib
import shutil
import imageio
import zipfile
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__, static_url_path="/")

CORS(app)  # 解决跨域问题
basedir = os.path.abspath(os.path.dirname(__file__))  # 使用绝对路径
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# 自动重载模板文件
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# 设置静态文件缓存过期时间
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

app.secret_key = 'secret_key'  # 设置表单交互密钥

weights_path = "../FZZD-model/ResNet34_1.pth"
weights_x_path = "../FZZD-model/ResNet34_test.pth"
class_json_path = "./class_indices.json"
class_json_x_path = "./class_indices_x.json"
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(class_json_path), "class json path does not exist..."

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = resnet34(num_classes=3).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model_x = resnet34(num_classes=4).to(device)
model_x.load_state_dict(torch.load(weights_x_path, map_location=device))

model.eval()
model_x.eval()

json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)
json_file = open(class_json_x_path, 'rb')
class_indict_x = json.load(json_file)

IMAGE_INPUT_TENSOR = 'Placeholder:0'
TRAINING_PH_TENSOR = 'is_training:0'
FINAL_CONV_TENSOR = 'resnet_model/block_layer4:0'
CLASS_PRED_TENSOR = 'ArgMax:0'
CLASS_PROB_TENSOR = 'softmax_tensor:0'
LOGITS_TENSOR = 'resnet_model/final_dense:0'
CLASS_NAMES = ('Normal', 'Pneumonia', 'COVID-19')
META_NAME = 'model.meta'
CKPT_NAME = 'model'
meta_file = os.path.join('./', META_NAME)
ckpt = os.path.join('./', CKPT_NAME)

unet = keras.models.load_model('../FZZD-model/unetmodel.h5', compile=False)
unet.summary()

dice_coef_loss = -0.8332
iou = 0.7186
dice_coef = 0.8332
BZDMD = keras.models.load_model('../FZZD-model/unet_lung_seg.hdf5',
                                custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou
                                    , 'dice_coef': dice_coef})
BZDMD.summary()

# 1、实例化登录管理对象
login_manager = LoginManager()

# 参数配置
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
login_manager.login_message = 'Access denied.'

login_manager.init_app(app)  # 初始化应用


@app.route('/user', methods=["POST", "GET"])
@login_required
def user():
    if request.method == 'POST':
        try:
            id = current_user.get_id()
            user = User.query.filter(User.id == id).all()
            for u in user:
                username = u.username
                mail = u.mail
                text = request.form.get('text')
                fk = FadeBack()
                fk.username = username
                fk.mail = mail
                fk.text = text
                db.session.add(fk)
                db.session.commit()
                return render_template('user.html', msg='提交成功！')
        except Exception as e:
            return render_template('user.html', msg='提交失败！-' + str(e))
    return render_template('user.html')


@app.route('/feedback', methods=["POST", "GET"])
@login_required
def feedback():
    if request.method == 'POST':
        try:
            id = current_user.get_id()
            user = User.query.filter(User.id == id).all()
            for u in user:
                username = u.username
                mail = u.mail
                text = request.form.get('text')
                fk = FadeBack()
                fk.username = username
                fk.mail = mail
                fk.text = text
                db.session.add(fk)
                db.session.commit()
                return render_template('feedback.html', msg='提交成功！')
        except Exception as e:
            return render_template('feedback.html', msg='提交失败！-' + str(e))
    return render_template('feedback.html')


@app.route('/userLogInfo', methods=["POST", "GET"])
@login_required
def userLogInfo():
    info = {}
    lists = []

    routers = {'resultCT': 'CT图像类别预测', 'resultXray': 'X光片类别预测',
                   'focalpoint_res': '标记CT图像COVID-19病灶点', 'heatmap_res': '绘制CT图像热力图',
                   'segXray_res': '绘制X光片肺部标记图', 'pngtonii_res': 'png堆叠转换nii文件',
                   'niitopng_res': 'nii文件转化为png格式'}
    try:
        if request.method == 'POST':
            id = current_user.get_id()
            userlogs = UserLog.query.filter(UserLog.userId == id).all()
            for userlog in userlogs:
                res = {}
                # res['userName'] = userlog.username
                res['id'] = userlog.id
                res['useType'] = routers[userlog.router]
                res['userInput'] = userlog.userInput
                res['resType'] = userlog.resType
                res['result'] = userlog.result
                res['dateTime'] = userlog.dateTime
                lists.append(res)
            info['result'] = lists
            info['msg'] = 'success'
    except Exception as e:
        info['err'] = str(e)
    return jsonify(info)


@app.route('/logStatus', methods=["POST", "GET"])
@login_required
def log_status():
    info = {}
    res = {}
    routers = {'resultCT': 'CT图像类别预测', 'resultXray': 'X光片类别预测',
               'focalpoint_res': '标记CT图像COVID-19病灶点', 'heatmap_res': '绘制CT图像热力图',
               'segXray_res': '绘制X光片肺部标记图', 'pngtonii_res': 'png堆叠转换nii文件',
               'niitopng_res': 'nii文件转化为png格式'}
    try:
        id = request.args.get('id')
        userlog = UserLog.query.filter(UserLog.id == id).all()
        # res['userName'] = userlog.username
        res['id'] = userlog[0].id
        res['useType'] = routers[userlog[0].router]
        res['userInput'] = userlog[0].userInput
        res['resType'] = userlog[0].resType
        res['result'] = json.loads(userlog[0].result)
        res['dateTime'] = userlog[0].dateTime
        info['result'] = res

        if userlog[0].resType == 'text':
            info['router'] = 'logStatus2.html'
        else:
            info['router'] = 'logStatus.html'
    except Exception as e:
        info['err'] = str(e)
    return render_template(info['router'], info=info)


# 3、加载用户, login_required 需要查询用户信息
@login_manager.user_loader
def user_loader(user_id: str):
    """
    [注意] 这里的user_id类型是str
    :param user_id:
    :return:
    """
    if User.query.filter_by(id=int(user_id)) is not None:
        curr_user = User()
        curr_user.id = user_id
        return curr_user


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        repassword = request.form.get('repassword')
        mail = request.form.get('mail')
        user = User.query.filter(User.username == username).all()
        if len(user) > 0:
            return render_template('register.html', msg='用户名已存在！')
        user1 = User.query.filter(User.mail == mail).all()
        if len(user1) > 0:
            return render_template('register.html', msg='邮箱已存在！')
        if password == repassword:
            # 注册用户
            user = User()
            user.username = username
            # 使用自带的函数实现加密：generate_password_hash
            user.password = hashlib.sha256(password.encode('utf-8')).hexdigest()
            user.mail = mail
            # 添加并提交
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # 关键  select * from user where username='xxxx';
        new_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
        # 查询
        user_list = User.query.filter_by(username=username)

        for u in user_list:
            # 此时的u表示的就是用户对象
            if u.password == new_password:
                login_user(u)
                return redirect(url_for('index'))
        else:
            return render_template('login.html', msg='用户名或者密码有误！')

    return render_template('login.html')


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))


def create_session():
    """Helper function for session creation"""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    return sess


def load_graph(meta_file):
    """Creates new graph and session"""
    graph = tf.Graph()
    with graph.as_default():
        # Create session and load model
        sess = create_session()

        # Load meta file
        saver = tf.compat.v1.train.import_meta_graph(meta_file, clear_devices=True)
    return graph, sess, saver


def load_ckpt(ckpt, sess, saver):
    """Helper for loading weights"""
    # Load weights
    if ckpt is not None:
        saver.restore(sess, ckpt)


graph, sess, saver = load_graph(meta_file)
with graph.as_default():
    load_ckpt(ckpt, sess, saver)


def make_gradcam_graph(graph):
    """Adds additional ops to the given graph for Grad-CAM"""
    with graph.as_default():
        # Get required tensors
        final_conv = graph.get_tensor_by_name(FINAL_CONV_TENSOR)
        logits = graph.get_tensor_by_name(LOGITS_TENSOR)
        preds = graph.get_tensor_by_name(CLASS_PRED_TENSOR)

        # Get gradient
        top_class_logits = logits[0, preds[0]]
        grads = tf.gradients(top_class_logits, final_conv)[0]

        # Comute per-channel average gradient
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    return final_conv, pooled_grads


final_conv, pooled_grads = make_gradcam_graph(graph)


def run_gradcam(final_conv, pooled_grads, sess, image):
    """Creates a Grad-CAM heatmap"""
    with graph.as_default():
        # Run model to compute activations, gradients, predictions, and confidences
        # 运行模型以计算激活、梯度、预测和置信度
        final_conv_out, pooled_grads_out, class_pred, class_prob = sess.run(
            [final_conv, pooled_grads, CLASS_PRED_TENSOR, CLASS_PROB_TENSOR],
            feed_dict={IMAGE_INPUT_TENSOR: image, TRAINING_PH_TENSOR: False})
        final_conv_out = final_conv_out[0]
        class_pred = class_pred[0]
        class_prob = class_prob[0, class_pred]

        # Compute heatmap as gradient-weighted mean of activations
        # 将热图计算为激活的梯度加权平均值
        for i in range(pooled_grads_out.shape[0]):
            final_conv_out[..., i] *= pooled_grads_out[i]
        heatmap = np.mean(final_conv_out, axis=-1)

        # Convert to [0, 1] range
        # 转换为[0，1]范围
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        # Resize to image dimensions
        # 调整到图像尺寸
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))

    return heatmap, class_pred, class_prob


def run_inference(graph, sess, images, batch_size=1):
    """Runs inference on one or more images"""
    # Create feed dict
    feed_dict = {TRAINING_PH_TENSOR: False}

    # Run inference
    with graph.as_default():
        classes, confidences = [], []
        num_batches = int(np.ceil(images.shape[0] / batch_size))
        for i in range(num_batches):
            # Get batch and add it to the feed dict
            feed_dict[IMAGE_INPUT_TENSOR] = images[i * batch_size:(i + 1) * batch_size, ...]

            # Run images through model
            preds, probs = sess.run([CLASS_PRED_TENSOR, CLASS_PROB_TENSOR], feed_dict=feed_dict)

            # Add results to list
            classes.append(preds)
            confidences.append(probs)

    classes = np.concatenate(classes, axis=0)
    confidences = np.concatenate(confidences, axis=0)

    return classes, confidences


def stacked_bar(ax, probs):
    """Creates a stacked bar graph of slice-wise predictions"""
    x = list(range(probs.shape[0]))
    width = 0.8
    ax.bar(x, probs[:, 0], width, color='g')
    ax.bar(x, probs[:, 1], width, bottom=probs[:, 0], color='r')
    ax.bar(x, probs[:, 2], width, bottom=probs[:, :2].sum(axis=1), color='b')
    ax.set_ylabel('Confidence')
    ax.set_xlabel('Slice Index')
    ax.set_title('Class Confidences by Slice')
    ax.legend(CLASS_NAMES, loc='upper right')


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # plt.imshow(image)
    # plt.show()
    # if image.mode != "L":
    #     raise ValueError("上传图片不是灰度图...")
    return my_transforms(image).to(device)


def transform_image_x(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # plt.imshow(image)
    # plt.show()
    # if image.mode != "L":
    #     raise ValueError("上传图片不是灰度图...")
    return my_transforms(image).to(device)


def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        tensor = torch.unsqueeze(tensor, dim=0)
        output = torch.squeeze(model(tensor.to(device)))
        pre = torch.softmax(output, dim=0)
        # predict_cla = torch.argmax(predict).numpy()
        template = "class: {:<15}\tprobability: {:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(pre)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


def get_prediction_x(image_bytes):
    try:
        tensor = transform_image_x(image_bytes=image_bytes)
        tensor = torch.unsqueeze(tensor, dim=0)
        output = torch.squeeze(model_x(tensor.to(device)))
        pre = torch.softmax(output, dim=0)
        # predict_cla = torch.argmax(predict).numpy()
        template = "class:{:<15}\tprobability:{:.3f}"
        index_pre = [(class_indict_x[str(index)], float(p)) for index, p in enumerate(pre)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


def judgecam(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "L":
        raise ValueError("上传图片不是灰度图...")
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)


def predict_unet(images):
    pred = unet.predict(images)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return pred


def zip_files(dir_path, zip_path):
    """
    :param dir_path: 需要压缩的文件目录
    :param zip_path: 压缩后的目录
    :return:
    """
    with zipfile.ZipFile(zip_path + '.zip', "w", zipfile.ZIP_DEFLATED) as f:
        for root, _, file_names in os.walk(dir_path):
            for filename in file_names:
                f.write(os.path.join(root, filename), filename)


@app.cli.command()
def create():
    db.drop_all()
    db.create_all()
    User.init_db()

def saveImage(image):
    """
    保存前端用户上传的图片
    """
    # 获取当前时间戳
    nowTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    path = 'static/data/' + nowTime + '.jpg'
    with open(path, 'wb') as f:
        f.write(image)
    f.close()
    return 'data/' + nowTime + '.jpg'


def saveUserLog(router, resType, userInput, output):
    # try:
    userId = current_user.get_id()
    user = User.query.filter(User.id == userId).all()
    userlog = UserLog()
    userlog.userId = userId
    userlog.username = user[0].username
    userlog.router = router
    userlog.resType = resType
    userlog.userInput = userInput
    userlog.result = json.dumps(output)
    db.session.add(userlog)
    db.session.commit()
    # except Exception as e:
    #     print(e)
    #     return str(e)
    return True


@app.route("/index", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    return render_template("predict.html")


@app.route("/resultCT", methods=["GET", "POST"])
@login_required
@torch.no_grad()
def result():
    image = request.files["file"]
    img_bytes = image.read()
    savePath = saveImage(img_bytes)

    info = get_prediction(image_bytes=img_bytes)

    saveUserLog('resultCT', 'text', savePath, info['result'])
    return jsonify(info)  # json格式传至前端


@app.route("/preCT", methods=["GET", "POST"])
@login_required
def preCT():
    return render_template("preCT.html")


@app.route("/resultXray", methods=["GET", "POST"])
@login_required
@torch.no_grad()
def resultXray():
    image = request.files["file"]
    img_bytes = image.read()
    savePath = saveImage(img_bytes)

    info = get_prediction_x(image_bytes=img_bytes)
    saveUserLog('resultXray', 'text', savePath, info['result'])
    return jsonify(info)  # json格式传至前端


@app.route("/preXray", methods=["GET", "POST"])
@login_required
def preXray():
    return render_template("preXray.html")


@app.route("/multiplepreCT", methods=["GET", "POST"])
@login_required
def multiplepreCT():
    return render_template("multiplepreCT.html")


@app.route("/multiplepreXray", methods=["GET", "POST"])
@login_required
def multiplepreXray():
    return render_template("multiplepreXray.html")


@app.route("/segmentation", methods=['GET', 'POST'])
@login_required
def segmentation():
    return render_template('segmentation.html')


@app.route("/heatmap_res", methods=["GET", "POST"])
@login_required
def heatmap_res():
    return_info = {}
    try:
        outputPath = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + '.png'
        image_file = request.files["file"]
        img_bytes = image_file.read()
        savePath = saveImage(img_bytes)
        # image = judgecam(image_bytes=img_bytes)
        # image = Image.open(io.BytesIO(img_bytes))
        # image = cv2.imdecode(io.BytesIO(img_bytes), cv2.IMREAD_GRAYSCALE)
        image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(np.asarray(image), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(np.stack((image, image, image), axis=-1), axis=0)
        # Run Grad-CAM
        heatmap, class_pred, class_prob = run_gradcam(
            final_conv, pooled_grads, sess, image)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(hspace=0.01)
        ax[0].imshow(image[0])
        plt.suptitle('Predicted Class: {} ({:.3f} confidence)\n'.format(CLASS_NAMES[class_pred], class_prob, ))
        ax[1].imshow(image[0])
        ax[1].imshow(heatmap, cmap='jet', alpha=0.4)
        plt.savefig("./static/data/" + outputPath, dpi=200)
        return_info["result"] = 'data/' + outputPath

        # 保存用户操作日志到数据库
        saveUserLog('heatmap_res', 'image', savePath, return_info['result'])
    except Exception as e:
        return_info['err'] = str(e)
    return return_info


@app.route("/heatmap", methods=["GET", "POST"])
@login_required
def heatmap():
    return render_template("heatmap.html")


@app.route("/segXray_res", methods=["GET", "POST"])
@login_required
def segXray_res():
    return_info = {}
    try:
        outputPath = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + '.png'
        images = []
        IMAGE_SIZE = 256
        image_file = request.files["file"]
        img_bytes = image_file.read()
        savePath = saveImage(img_bytes)

        image = Image.open(io.BytesIO(img_bytes))
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        image = image_utils.img_to_array(image)

        image = np.mean(image, axis=-1) / 255.0
        images.append(image)
        image = np.array(images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        pred = predict_unet(image)
        plt.subplot(1, 2, 1)
        plt.imshow(image[0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(image[0], cmap='gray', interpolation='none')
        plt.imshow(pred[0], cmap='Spectral_r', alpha=0.3)
        plt.savefig("./static/data/" + outputPath, dpi=200)
        plt.close()
        return_info["result"] = 'data/' + outputPath

        # 保存用户操作日志到数据库
        saveUserLog('segXray_res', 'image', savePath, return_info['result'])
    except Exception as e:
        return_info['err'] = str(e)
    return return_info


@app.route("/segXray", methods=["GET", "POST"])
@login_required
def segXray():
    return render_template("segXray.html")


@app.route("/focalpoint_res", methods=["GET", "POST"])
@login_required
def focalpoint_res():
    return_info = {}
    try:
        outputPath = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + '.png'
        image_file = request.files["file"]
        img_bytes = image_file.read()
        savePath = saveImage(img_bytes)

        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
        img = cv2.resize(img, (512, 512))
        image = img
        img = img / 255
        img = img[np.newaxis, :, :, :]
        pred = BZDMD.predict(img)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.subplot(1, 3, 2)
        plt.imshow(np.squeeze(pred) > .5, cmap='gray')
        plt.title('Prediction')
        plt.subplot(1, 3, 3)
        plt.imshow(image, cmap='gray',
                   interpolation='none')
        plt.imshow(np.squeeze(pred) > .5, cmap='Spectral_r', alpha=0.5)
        plt.title('Actual')
        plt.savefig("static/data/" + outputPath, dpi=300)
        plt.close()
        return_info["result"] = 'data/' + outputPath

        # 保存用户操作日志到数据库
        saveUserLog('focalpoint_res', 'image', savePath, return_info['result'])
    except Exception as e:
        return_info['err'] = str(e)
    return return_info


@app.route("/focalpoint", methods=["GET", "POST"])
@login_required
def focalpoint():
    return render_template("focalpoint.html")


@app.route("/conversion", methods=["GET", "POST"])
@login_required
def conversion():
    return render_template("conversion.html")


@app.route("/pngtonii_res", methods=["GET", "POST"])
@login_required
def pngtonii_res():
    return_info = {}
    try:
        nowTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        outputPath = nowTime + '.nii'
        savePath = 'static/data/' + nowTime
        if not os.path.exists(savePath):
            os.mkdir(savePath)

        imgs = []
        for img in request.files:
            img_bytes = request.files[img].read()
            fileName = request.files[img].filename.split('/')[1]
            with open(savePath + '/' + fileName, 'wb') as f:
                f.write(img_bytes)
            f.close()
            image = Image.open(io.BytesIO(img_bytes))
            img = np.array(image)
            imgs.append(img)

        # 打包成 zip
        zip_files(savePath, savePath)
        shutil.rmtree(savePath)

        imgnii = np.stack(imgs, axis=0)
        nii = sitk.GetImageFromArray(imgnii)
        sitk.WriteImage(nii, './static/data/' + outputPath)
        return_info['result'] = 'data/' + outputPath
        return_info['msg'] = 'success'

        # 保存用户操作日志到数据库
        saveUserLog('pngtonii_res', 'nii', 'data/' + nowTime + '.zip', return_info['result'])
    except Exception as e:
        return_info['err'] = str(e)
    return return_info


@app.route("/pngtonii", methods=["GET", "POST"])
@login_required
def pngtonii():
    return render_template("pngtonii.html")


@app.route("/niitopng_res", methods=["GET", "POST"])
@login_required
def niitopng_res():
    return_info = {}
    try:
        nii_file = request.files.get('file')

        nowTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        file = nowTime + '.nii'
        fname = file.replace('.nii', '')
        img_f_path = os.path.join('static/data/', fname)
        zip_path = 'static/data/' + nowTime

        with open('static/data/' + file, 'wb') as f:
            f.write(nii_file.read())
        f.close()

        img = nib.load('static/data/'+file)
        img_fdata = img.get_fdata()

        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)
        (x, y, z) = img.shape
        for i in range(z):  # z是图像的序列
            silce = img_fdata[:, :, i]  # 选择哪个方向的切片都可以
            imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), silce)  # 保存图像

        # 打包成 zip
        zip_files(img_f_path, zip_path)
        shutil.rmtree(img_f_path)

        return_info['result'] = 'data/' + nowTime + '.zip'
        return_info['msg'] = 'success'

        # 保存用户操作日志到数据库
        saveUserLog('niitopng_res', 'zip', 'data/' + file, return_info['result'])
    except Exception as e:
        return_info['err'] = str(e)
    return return_info


@app.route("/niitopng", methods=["GET", "POST"])
@login_required
def niitopng():
    return render_template("niitopng.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=98765)
