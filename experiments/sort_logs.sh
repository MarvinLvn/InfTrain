#!/bin/bash




for file in logs/*.out
do 
   
    # check that file is older than today otherwise can break if logs currently being created
    if [[ $(find "$file" -mtime +0 -print) ]]; then
        date=$(stat -c '%y' $file | cut -d' ' -f1 )
        year=$(echo $date | cut -d'-' -f1)
        month=$(echo $date | cut -d'-' -f2)

        ### create the variable for store directory name
        STOREDIR=logs/${year}/${month}/${date}

        # echo $file

        if [ -d ${STOREDIR} ]         ### if the directory exists
        then
            mv ${file} ${STOREDIR}    ### move the file
        else                          ### the directory doesn't exist
            mkdir -p ${STOREDIR}         ### create it
            mv ${file} ${STOREDIR}    ### then move the file
        fi                            ### close if statement
    fi
done                   
