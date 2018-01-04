find . -type f -exec bash -c 'file -bi "$1" | grep -q image/jpeg || rm "$1"' none {} \;
