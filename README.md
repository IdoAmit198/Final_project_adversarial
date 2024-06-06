# Final_project_adversarial

## Description
In this work, we assessed the contribution of the adaptive setting introduced in the [CAT paper](https://arxiv.org/abs/2002.06789), proposed additional settings, and researched the effect those have on an adversarially trained model.

We adversarially train on the train set of Cifar-10 and evaluate robustness on the test set for a wide range of epsilon values.

## Install


## How to run
Generally, the main script is the `run.py`. Both training and evaluation are done using this script.
### Adversarial train
To train a robust model, one can run the `run.py` script.

1. For example, for regular adversarial training as done in [Madry et al., 2017](https://arxiv.org/abs/1706.06083), run the script as follows:
```
python run.py --train_method train --model_name resnet18 --max_epsilon 8
```
It will train a ResNet-18 model from scratch with a PGD-10 attack, using a perturbation size of 8/255.

2. let us review another example. For adversarial training in the adaptive setting, as was suggested in CAT, using a max epsilon of 16 and PGD-20 instead of 10, run the following:
```
python run.py --train_method adaptive --model_name resnet18 --max_epsilon 16 --pgd_num_steps 20
```

3. If you want to train a model in the Re-Introduce setting as we proposed in our work, adjust the `train_method` flag again. Notice, that we can also modify the increment of epsilons' value, which we use in the adaptive and e-Introduce settings. Run the following to train a model with our proposed Re-Introduce setting with an epsilon step size of 0.002 instead of the default 0.005:
```
python run.py --train_method re_introduce --epsilon_step_size 0.002
```
4. Lastly, we also proposed the incorporation of the Target-Agnostic loss term instead of the Cross-Entropy. To train a model using this loss, add the flag `agnostic_loss`. For instance, to train a model with that loss in the Re-Introduce setting, run the following:
```
python run.py --train_method re_introduce --agnostic_loss
```
For further information, run the script with the `help` flag to see the full list of arguments and how to use them:
```
python run.py --help
```
**Notice:** When using the SLURM system, one can easily use our bash script, `scripts/<method_script>`, to run a job with the wanted training scheme. For example, to train in the adaptive setting, use the `scripts/adaptive_baseline.sh` and adjust the required flags as explained above.

### Evaluate a trained model
As explained in the pdf report, the evaluation is done for a range of epsilons to test robustness for a range of attacks' strengths. To run an evaluation, one should pass the `eval_epsilons` flag. Additionally, we recommend adjusting the rest of the relevant arguments, which are:
    1. eval_epsilon_max.
    2. eval_model_path.
    3. eval_uncertainty
Each argument is documented in the `utils/args.py` file.

For example, to run an evaluation, using a model in the path `saved_models/resnet18/seed_42/train_method_re_introduce/agnostic_loss_True/max_epsilon_32.pth` for the range of epsilons from 0 to 32, one can run the following:
```
python run.py --eval_epsilons \
              --eval_model_path saved_models/resnet18/seed_42/train_method_re_introduce/agnostic_loss_True/max_epsilon_32.pth
```
**Notice:** For evaluation purposes in the SLURM system, there is a bash script named `scripts/eval_epsilons.sh`. Modify the passed arguments and you're ready to go.

### Plot results
To visualize the results, one can use the `plots.py` script. The script does not require passing arguments and can be run as follows:
```
python plots.py
```
