import time
import sys
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import xml.etree.ElementTree as ET
from PIL import Image
import io
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
import random
import binascii
from phe import paillier
import tenseal as ts

class TenSealEnc():
    def __init__(self, kappa):
        self.context = None
        self.kappa = kappa

    def encrypt(self, X):
        # encrypted vectors
        enc1 = ts.ckks_vector(self.context, X)
        #enc2 = ts.ckks_vector(self.context, Y)
        return enc1

    def decrypt(self, enc1):
        decry = enc1.decrypt()
        return decry

    def generate_keys(self):
        if self.kappa ==512:
            self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                                 coeff_mod_bit_sizes=[60, 40, 40, 60])
        if self.kappa == 1024:
            self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384,
                                 coeff_mod_bit_sizes=[60, 40, 40, 60, 60, 40, 40, 60])
        if self.kappa == 2048:
            self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768,
                                 coeff_mod_bit_sizes=[60, 40, 40, 60, 60, 40, 40, 60, 60, 40, 40, 60])
        if self.kappa == 4096:
            self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=65536,
                                 coeff_mod_bit_sizes=[60, 40, 40, 60, 60, 40, 40, 60, 60, 40, 40, 60, 60, 40, 40, 60])
        self.context.generate_galois_keys()
        self.context.global_scale = 2 ** 40

class RSA:
    def __init__(self):
        self.sk = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.pk = self.sk.public_key()

    def sign_digest(self, digest):

        signature = self.sk.sign(
            digest,
            padding=padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            algorithm=hashes.SHA256()
        )
        return signature

    def verify_digest_signature(self, digest, signature):

        try:
            self.pk.verify(
                signature,
                digest,
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                algorithm=hashes.SHA256()
            )
            return True
        except Exception:
            return False

    def sign_image(self, image_data):

        hasher = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hasher.update(image_data)
        hash_value = hasher.finalize()

        signature = self.sk.sign(
            hash_value,
            padding=padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            algorithm=hashes.SHA256()
        )
        return signature

    def verify_image_signature(self, image_data, signature):

        hasher = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hasher.update(image_data)
        hash_value = hasher.finalize()


        try:
            self.pk.verify(
                signature,
                hash_value,
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                algorithm=hashes.SHA256()
            )
            return True
        except Exception:
            return False

    def sign_text(self, text):

        text_bytes = text.encode('utf-8')


        hasher = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hasher.update(text_bytes)
        hash_value = hasher.finalize()

        signature = self.sk.sign(
            hash_value,
            padding=padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            algorithm=hashes.SHA256()
        )
        return signature

    def verify_text_signature(self, text, signature):

        text_bytes = text.encode('utf-8')

        hasher = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hasher.update(text_bytes)
        hash_value = hasher.finalize()

        try:
            self.pk.verify(
                signature,
                hash_value,
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                algorithm=hashes.SHA256()
            )
            return True
        except Exception:
            return False



class AESEncrypt:
    def __init__(self):
        seed = 1
        random.seed(seed)
        self.Ks = bytes([random.randint(0, 255) for _ in range(16)])

    def aes_encrypt(self,data):
        cipher = AES.new(self.Ks, AES.MODE_ECB)
        #ciphertext = cipher.encrypt(pad(data, AES.block_size))
        ciphertext = cipher.encrypt(pad(data, AES.block_size))
        return ciphertext

    def encrypt(self, img_path=None, img_output_path=None, txt_path=None, txt_output_path=None):
        if img_path is not None:
            with open(img_path, 'rb') as file:
                image_data = file.read()
            ciphertext = self.aes_encrypt(image_data)

            with open(img_output_path, 'wb') as file:
                file.write(ciphertext)
        if txt_path is not None:
            xml_text = []
            with open(txt_path, "r", encoding="utf-8") as file:
                xml_text = file.read()

                xml_text = xml_text.replace("&", "&amp;")


            tree = ET.ElementTree(ET.fromstring(xml_text))
            root = tree.getroot()
            text_data = None

            for item in root.findall('.//text'):
                text_data = item.text

            text_data = text_data[1:-1]
            ciphertext2 = self.aes_encrypt(text_data.encode('utf-8'))

            with open(txt_output_path, 'wb') as file:
                file.write(ciphertext2)

        return True

    def aes_decrypt(self, ciphertext):
        cipher = AES.new(self.Ks, AES.MODE_ECB)
        decrypted_data = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted_data

    def decrypt(self, img_path=None, img_output_path=None, txt_path=None, txt_output_path=None):
        if img_path is not None:
            with open(img_path, 'rb') as file:
                ciphertext = file.read()

            decrypted_data = self.aes_decrypt(ciphertext)

            with open(img_output_path, 'wb') as file:
                file.write(decrypted_data)

        if txt_path is not None:
            with open(txt_path, 'rb') as file:
                ciphertext2 = file.read()

            decrypted_data2 = self.aes_decrypt(ciphertext2).decode('utf-8')

            with open(txt_output_path, 'w', encoding='utf-8') as file:
                file.write(decrypted_data2)

        return True

if __name__ == '__main__':

    public_key, private_key = paillier.generate_paillier_keypair(n_length=2048)
    X = np.random.randint(0, 100, 100)

    enc = 0
    dec = 0

    for x in X:
        time1 = time.time()
        encX = public_key.encrypt(int(x))
        enc += time.time() - time1
        time2 = time.time()
        decX = private_key.decrypt(encX)
        dec += time.time() - time2

    print("enc:{}".format(enc/100))
    print("dec:{}".format(dec / 100))