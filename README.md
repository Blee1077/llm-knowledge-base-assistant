# llm-knowledge-base-assistant

https://gxjdinj3ddxkgkdfnu2eft6zd40bsbjt.lambda-url.eu-west-2.on.aws/

## To run
1. Ensure CDK is installed
```
$ npm install -g aws-cdk
```

2. Create a Python virtual environment
```
$ python3 -m venv .venv
```

3. Activate virtual environment

_On MacOS or Linux_
```
$ source .venv/bin/activate
```

_On Windows_
```
% .venv\Scripts\activate.bat
```

4. Install the required dependencies from root folder.

```
$ pip install -r ./requirements.txt
```

5. Synthesize (`cdk synth`) or deploy (`cdk deploy`) the example

```
$ cdk deploy
```

## To dispose of the stack afterwards:

```
$ cdk destroy
```
