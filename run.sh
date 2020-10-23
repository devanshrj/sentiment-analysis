# function to check if a python package is installed
function package_exist() {
    package=$1
    if pip freeze | grep $package=; then
        echo "$package is installed"
    else
        echo "$package is not installed"
        pip install "$package"
    fi
}

# check torchtext
package_exist torchtext

# check transformers
package_exist transformers

# execute run.py
echo "Executing run.py"
python -W ignore run.py -n "$1"
