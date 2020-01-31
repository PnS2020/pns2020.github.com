---
layout: default
---

# Git Crash Course

We strongly recommend this [online Git tutorial](https://try.github.io/).
You can learn how to use `Git` in 15 steps!

## Why do I need Git? And what is Git?

[Git](https://git-scm.com/) is a source code version control system.
The main purpose of having such a system is to maintain the code changes.
With Git, you can track the history, save yourself from a post-catastrophic situation (e.g, the precious laptop is destroyed because you ate in front of it), integrate the work from others, make a copy of existing work along with all the history, and many other useful features.

## If you are not a fan of terminal

The following content of this page navigates yourself to be familiar with
the command of Git. However, you may not be a fan of terminal and that
is absolutely fine. You can download the [GitHub Desktop](https://desktop.github.com/) client here and have fun!

## Clone, commit and push!

Suppose you have a directory that hosts your project, you can initialize a Git environment by the following command:

```
$ git init
```

The above command only initializes the local Git environment where you can update your project and preserve your history using Git's commands. However, to backup your projects in some remote server (e.g., GitHub), you will have to configure the `remote`. We will skip this part of configuration for now.

The other way of working on a project is to `clone`. This is in fact the most common situation where you have a project on the remote server and you would like to work on it with your own personal computer. To do so, you need to download the project using `git clone` command:

```
$ git clone https://github.com/PnS2020/git-hello-world
```

Note that `git clone` not only downloads the project, but also configure the remote connections. Hence, if you have any modifications (e.g., commits) to the project, you can now `push` to the remote server.

Following lines demonstrate how you can push your first file:

```bash
$ cd git-hello-world  # navigate into the folder
$ touch hello-world.txt  # create a file named hello-world.txt
$ echo "Hello World!" > hello-world.txt  # append the string "Hello World!" to the document
$ git add hello-world.txt  # Add (stage) the file so that git can track it
$ git commit -m "first commit"  # make the commit, now this change is a part of history
$ git push origin master  # push it to master branch if there is a remote server
```

Normally, Git first tracks the difference between the current version and the most recent version in the history (you can see the changes with `git status`), then you have to make a decision if you want to stage these changes through `git add`. After you staged the changes, you would need to record this change to the history via `git commit`.

## Pulling, Branching, Fetching and Merging

Say if you made some changes at your working computer and want to continue your work on your own PC (I know), however, the version of code on your PC may not agree with the latest version. In this case, you will need to pull the changes from the remote server so that you can catch up:

```
$ git pull origin master
```

So there is another "pulling" command `fetch` in `git`. In the simplest terms, `git pull` does a `git fetch` followed by a `git merge`. However, we found that `git fetch` is particularly useful when you try to pull all the remote branches to local system:

```
$ git fetch
```

Branching is an important feature of Git. Initially, you have a default branch `master`. Say you want to add some feature / fix a bug to your project (e.g., let it say something funny, make sure it does not curse anyone). Generally, the idea is to make the `master` branch as stable as possible. You want to avoid any untested changes. So the way you do it, is to create a branch where this branch has all the previous history. Your work on this branch will not affect the master branch.
After you properly tested this new feature, you can then merge the changes to master branch.

To create a new branch, use:

```
$ git checkout -b new-feature
```

Switch to another branch, use:

```
$ git checkout new-feature
```

After you've done coding and testing the new features in the branch `new-feature`. You would merge the changes to the master branch. The way you do it is through following commands:

```branch
$ git checkout master  # switch to master branch
$ git merge new-feature  # merge the new-feature branch
```

Note that you would be able to automatically merge two branches together in most cases. However, if someone changed that same file in the mean time, then you will need to resolve the conflicts between different changes. In fact, this situation happens all the time if you are working on a very active project.

## Respect the flow!

GitHub encourages a workflow paradigm while developing a project.
The idea is to keep the `master` branch stable, and any changes to the master branch has to be tested and approved via code review. The details of the
[GitHub Flow](https://guides.github.com/introduction/flow/) is here.

## Continuous Integration (Optional)

Usually, we will assume some requirements for running a project.
If all the requirements are installed, a user should be able to compile or run the project without a problem.

However, this assumption causes many problems because the programmer or the development team may not keep track of the requirement list. This assumption also makes the testing harder. So, to resolve this problem, we encourage people to use _Continuous Integration_ (CI), a system that configures a building environment that meets all the requirements and then run tests. In this way, the team will know the exact configuration for building a project, and any extra additions to the project will be notified.

The popular CI system for Linux and macOS is [Travis CI](https://travis-ci.org/) and for Windows is [AppVeyor](https://www.appveyor.com/).

## Further Readings

+ [Official Cheat Sheet](https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf)
+ [GitHub Guides](https://guides.github.com/)
+ [Online Git tutorial](https://try.github.io/)
+ [Learn Git with Bitbucket Cloud](https://www.atlassian.com/git/tutorials/learn-git-with-bitbucket-cloud)
