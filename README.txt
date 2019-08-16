
# == Setup to work in a forked repository ==
## Fork an upstream repository via github web ui
git clone your_forked_repository_url  # download a copy of your forked repo
git remote add upstream the_upstream_repository_url  # record from where your repo was forked
git branch your_new_branch  # create a new branch to work in
git checkout your_new_branch  # change your working tree to the new branch
## Shorthand for the prev 2 lines: git checkout -b your_new_branch

# == Check connections of your local repository ==
git remote -v  # report repositories to which your repository is linked
## "origin" is your forked hosted repo
## "upstream" is from where your hosted repo was forked
##
## Expect to see something like this if configured properly:
##     origin  https://github.com/TimoLarson/julia (fetch)
##     origin  https://github.com/TimoLarson/julia (push)
##     upstream        https://github.com/JuliaLang/julia (fetch)
##     upstream        https://github.com/JuliaLang/julia (push)

# == Check statut of your local repository ==
git status  # report info about your current working tree
## (e.g. what branch it is, pending updates)

# == Update forked master ==
git checkout master  # prepare the working tree for the merge below
git fetch upstream  # get updates from the upstream repo
git merge upstream/master  # incorporate the upstream updates into the master branch of your fork
git push  # update your hosted fork repo

# == Rebase branch on updated master ==
git rebase master # Note that this rewrites history locally
git push -f # Note that this rewrites history on the origin server

# == Create a linked working tree ==
git worktree add your_new_working_tree_path

# == Move (e.g. rename) a linked working tree ==
git worktree move old_working_tree_path new_working_tree_path

