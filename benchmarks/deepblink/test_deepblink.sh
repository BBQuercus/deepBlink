source "../envs.sh"

mLen=${#DB_LIST[@]}
dLen=${#DATA_LIST[@]}

if [ ! ${mLen} -eq ${dLen} ]; then
    echo "Length missmatch!"
    exit
fi

for ((i = 1; i < mLen + 1; i++)); do
    echo "Running ${DB_LIST[$i]} on ${DATA_LIST[$i]}"
    python deepblink_test.py -d ${DATA_LIST[$i]} \
        -o $DB_OUT -m ${DB_LIST[$i]}
done
