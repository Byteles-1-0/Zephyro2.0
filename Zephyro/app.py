# app.py
from dotenv import load_dotenv
load_dotenv()
from flask import Flask
from flask_compress import Compress
from flask_cors import CORS
from routes import bp as main_bp
from routes_predict import predict_bp


def create_app():
    app = Flask(__name__)
    Compress(app)  # ⬅️
    CORS(app)
    app.config['SECRET_KEY'] = 'supersecret'
    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)
    return app

app = create_app()

if __name__ == '__main__':
    #app.run(debug=True, port=5001)
    app.run(host="0.0.0.0", port=5001, debug=False)