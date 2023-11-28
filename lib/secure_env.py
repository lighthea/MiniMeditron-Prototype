import base64
import getpass
import re
from typing import Tuple
from os import environ
from Crypto.Protocol.KDF import scrypt
from Crypto.Cipher import AES

global password
password = None

SECURED_PREFIX = 'ENCRYPTED{'
SECURED_SUFFIX = '}'
REQUIRE_ENCRYPTION_PREFIX = 'DO_ENCRYPT{'
REQUIRE_ENCRYPTION_SUFFIX = '}'
VALID_VALUE_REGEX = r'[ \t\r\f\w;,\-+\\\/\=]*'
SEPARATOR = ';'
SCRYPT_PARAM_N = 2**16
SCRYPT_PARAM_R = 8

def read_secure_file(filename: str) -> str:
    # First open the file for readonly and read the whole file
    with open(filename, 'r') as f:
        content = f.read()

    # Iterate over the file to find some non-encrypted value
    pattern = re.compile(re.escape(REQUIRE_ENCRYPTION_PREFIX) + VALID_VALUE_REGEX + re.escape(REQUIRE_ENCRYPTION_SUFFIX))
    content, updated = _regex_replace(pattern, content, lambda x: _encrypt_value(x[len(REQUIRE_ENCRYPTION_PREFIX):-len(REQUIRE_ENCRYPTION_SUFFIX)]))
    # Find if we can still find some DO_ENCRYPT{ in the string
    if content.count(REQUIRE_ENCRYPTION_PREFIX) > 0:
        print("WARNING: It seems that some value didn't match the VALID_VALUE_REGEX and as such is not secure by this script")
        raise Exception("Invalid VALID_VALUE_REGEX matching")

    # If the content has been updated then update the file
    if updated:
        print('Update content of file {} with secured content'.format(filename))
        with open(filename, 'w') as f:
            f.write(content)
    
    # Perform the decryption
    pattern = re.compile(re.escape(SECURED_PREFIX) + VALID_VALUE_REGEX + re.escape(SECURED_SUFFIX))
    content, _ = _regex_replace(pattern, content, lambda x: _decrypt_value(x[len(SECURED_PREFIX):-len(SECURED_SUFFIX)]))

    # Finally return the decrypted secure content
    return content

def _regex_replace(regex: re.Pattern, string: str, transform) -> Tuple[str, bool]:
    newstring = ''
    start = 0
    updated = False
    for match in re.finditer(regex, string):
        end, newstart = match.span()
        newstring = string[start:end]
        newstring += transform(string[end:newstart])
        start = newstart
        updated = True
    newstring += string[start:]
    return newstring, updated

def _get_password() -> str:
    global password
    if password is None:
        if 'SECRET_PASSWORD' in environ:
            password = environ['SECRET_PASSWORD']
        else:
            password = getpass.getpass(prompt='Enter your password $ ')
    return password

def _decrypt_value(value: str):
    password = _get_password()

    # Retrieve the nonce, ciphertext and tag
    nonce, ciphertext, tag = [base64.b64decode(x) for x in value.split(SEPARATOR)]
    key = scrypt(password, '', 32, N=SCRYPT_PARAM_N, r=SCRYPT_PARAM_R, p=1)

    # Attempt to decrypt the plaintext
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)

    # Verify the result
    try:
        cipher.verify(tag)
        return plaintext.decode('utf-8')
    except ValueError:
        print('Incorrect password')
        raise Exception("Invalid Password Exception")

def _encrypt_value(value: str):
    password = _get_password()

    # No need for salt, the key is always used with nonce
    key = scrypt(password, '', 32, N=SCRYPT_PARAM_N, r=SCRYPT_PARAM_R, p=1)
    
    # Encrypt the password
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(value.encode('utf-8'))

    # Finally make the value
    value = SEPARATOR.join([base64.b64encode(x).decode('utf-8') for x in [nonce, ciphertext, tag]])
    return SECURED_PREFIX + value + SECURED_SUFFIX
