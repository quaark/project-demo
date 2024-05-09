# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from kfp import dsl

from mlrun import build_function, deploy_function, get_current_project, run_function
from mlrun.model import HyperParamOptions

funcs = {}
DATASET = "iris_dataset"
LABELS = "label"

in_kfp = True


@dsl.pipeline(name="Demo training pipeline", description="Shows how to use mlrun.")
def newpipeshort():

    project = get_current_project()

    # build our ingestion function (container image)
    builder = build_function("gen-iris")

    # run the ingestion function with the new image and params
    ingest = run_function(
        "gen-iris",
        name="get-data",
        handler="iris_generator",
        params={"format": "pq"},
        outputs=[DATASET],
    ).after(builder)
    print(ingest.outputs)

    # train with hyper-paremeters
    train = run_function(
        "auto-trainer",
        name="train",
        params={
            "label_columns": LABELS, 
            "train_test_split_size": 0.10,

            # for determinism, we need datasets to be split
            # evenly datasets will include examples from all classes
            "random_state": 7,
        },
        hyperparams={
            "model_class": [
                "sklearn.ensemble.RandomForestClassifier",
                "sklearn.linear_model.LogisticRegression",
                "sklearn.ensemble.AdaBoostClassifier",
            ]
        },
        hyper_param_options=HyperParamOptions(selector="max.accuracy"),
        inputs={"dataset": ingest.outputs[DATASET]},
        outputs=["model", "test_set"],
    )
    print(train.outputs)

    # deploy our model as a serverless function, we can pass a list of models to serve
    deploy = deploy_function(
        "serve",
        models=[{"key": f"{DATASET}:v1", "model_path": train.outputs["model"]}],
    )

