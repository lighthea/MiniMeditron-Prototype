scp -P 2244 root@127.0.0.1:MiniMeditron-Prototype/diff.patch diff.patch
git apply diff.patch
git add -A
git commit -m "$1"
git push
