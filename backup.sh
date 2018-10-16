#!/bin/sh

yes_or_no_while(){
    while true;do
        echo
        echo "Do you want to back up? Type yes or no."
        read answer
        case $answer in
            yes)
                echo -e "tyeped yes.\n"
                git add *.*
                git commit -m "Modified by MK."
                git push
                return 0
                ;;
            no)
                echo -e "tyeped no.\n"
                return 1
                ;;
            *)
                echo -e "cannot understand $answer.\n"
                ;;
        esac
done
}

git status
yes_or_no_while
