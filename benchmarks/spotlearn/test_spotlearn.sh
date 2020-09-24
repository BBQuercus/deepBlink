source "../envs.sh"

mLen=${#SL_LIST[@]}
dLen=${#DATA_LIST[@]}

if [ ! ${mLen} -eq ${dLen} ]; then
    echo "Length missmatch!"
    exit
fi

for ((i = 1; i < mLen + 1; i++)); do
    echo "Running ${SL_LIST[$i]} on ${DATA_LIST[$i]}"
    python spotlearn_test.py -d ${DATA_LIST[$i]} \
        -o $SL_OUT -m ${SL_LIST[$i]}
done
