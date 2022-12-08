# -*- coding: utf-8 -*-
import hashlib

from extension import db
from werkzeug.security import check_password_hash,  generate_password_hash
import hashlib
from flask_login import UserMixin


class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), nullable=False)
    mail = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(255), nullable=False)

    @staticmethod
    def init_db():
        rets = [
            [1, 'admin', '2092653757@qq.com', 'a1b2c3']
        ]
        for ret in rets:
            user = User()
            user.id = ret[0]
            user.username = ret[1]
            user.mail = ret[2]
            user.password = hashlib.sha256(ret[3].encode('utf-8')).hexdigest()
            db.session.add(user)
        db.session.commit()

    @classmethod
    def query_user_by_name(cls, username):
        for user in cls.users:
            if username == user['username']:
                return user

    @classmethod
    def query_user_by_id(cls, user_id):
        for user in cls.users:
            if user_id == user['id']:
                return user


class FK(db.Model):
    __tablename__ = 'feedback'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), nullable=False)
    mail = db.Column(db.String(255), nullable=False)
    text = db.Column(db.Text, nullable=False)
