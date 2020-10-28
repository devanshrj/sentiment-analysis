# function to check if a python package is installed
function package_exist() {
    package=$1
    if pip freeze | grep $package=; then
        echo "$package is installed"
    else
        echo "$package is not installed"
        pip install "$package"
        echo "$package is installed"
    fi
}

# check torchtext
echo "Check TorchText"
package_exist torchtext

# check transformers
echo "\nCheck Transformers"
package_exist transformers

# execute run.py
echo "\nExecuting train.py"
python -W ignore code/train.py -n "$1"
