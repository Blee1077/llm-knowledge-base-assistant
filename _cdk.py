import os
from pathlib import Path
from constructs import Construct
from aws_cdk import App, Stack, Environment, Duration, CfnOutput, aws_iam
from aws_cdk.aws_lambda import DockerImageFunction, DockerImageCode, FunctionUrlAuthType

# Environment
# CDK_DEFAULT_ACCOUNT and CDK_DEFAULT_REGION are set based on the
# AWS profile specified using the --profile option.
my_environment = Environment(account=os.environ["CDK_DEFAULT_ACCOUNT"], region=os.environ["CDK_DEFAULT_REGION"])

class KnowledgeBaseAssistDemo(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ##############################
        #       Lambda Function      #
        ##############################
        lambda_fn = DockerImageFunction(
            self,
            "AssetFunction",
            code=DockerImageCode.from_image_asset(str(Path.cwd()), file="Dockerfile"),
            memory_size=1024,
            timeout=Duration.minutes(3),
            environment={
                "HAYSTACK_TELEMETRY_ENABLED": "False"
            },
        )
        
        # Add HTTPS URL to access Gradio app via browser
        fn_url = lambda_fn.add_function_url(auth_type=FunctionUrlAuthType.NONE)
        CfnOutput(self, "functionUrl", value=fn_url.url)
        
        # Add S3 bucket access to lambda IAM role
        lambda_fn.add_to_role_policy(
            aws_iam.PolicyStatement(
                effect=aws_iam.Effect.ALLOW,
                actions=['s3:ListBucket', 's3:GetObject'],
                resources=['arn:aws:s3:::knowledge-base-assist-demo/*']
            )
        )

app = App()
rust_lambda = KnowledgeBaseAssistDemo(app, "KnowledgeBaseAssistDemo", env=my_environment)

app.synth()