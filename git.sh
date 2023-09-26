#!/bin/bash

rm -rf mediapipe9
git clone https://github.com/kuangzy2011/mediapipe9.git
cd mediapipe9
git init
git remote -v
git config --global user.name "kuangzy2011"
git config --global user.email "epay.kuangzy@126.com"
git pull origin main

cp -r ../mediapipe-0.9.1/* .
ls -l

git add bazel-bin
git add bazel-mediapipe-0.9.1
git add bazel-out
git add .bazelrc
git add bazel-testlogs
git add .bazelversion
git add build_android_examples.sh
git add BUILD.bazel
git add build_desktop_examples.sh
git add build_ios_examples.sh
git add CONTRIBUTING.md
git add Dockerfile
git add .dockerignore
git add docs
git add facemeshgpu.apk
git add .github
git add .gitignore
git add git.sh
git add handtrackinggpu.apk
git add handview.py
git add helloworld.apk
git add holistictrackinggpu.apk
git add LICENSE
git add MANIFEST.in
git add mediapipe
git add package.json
git add README.md
git add requirements.txt
git add setup_android_sdk_and_ndk.sh
git add setup_opencv.sh
git add setup.py
git add third_party
git add tsconfig.json
git add WORKSPACE
git add yarn.lock

sed -i 's/url = https:\/\/github.com\/kuangzy2011\/mediapipe9.git/url = https:\/\/kuangzy2011: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@github.com\/kuangzy2011\/mediapipe9.git/g' .git/config
cat .git/config
git status -s
git commit -m "Commit change to remote repository debug4"
git push -u origin main --force


