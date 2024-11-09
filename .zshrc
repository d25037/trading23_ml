# # linuxbrew
# export PATH="$PATH:/opt/nvim-linux64/bin"
# # eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

# TimeZone
export TZ="Asia/Tokyo"

# rust
export PATH="$HOME/.cargo/bin:$PATH"

# starship
eval "$(starship init zsh)"

# zoxide
eval "$(zoxide init zsh)"
export _ZO_FZF_OPTS="--preview 'eza --color=always --long --header --icons --classify --git --no-user {2..}' --height=70% --layout=reverse"

# eza
alias el='eza --all --long --header --icons --classify --git --no-user'
alias elm='eza --all --long --header --icons --classify --git --no-user --sort=modified'
alias elt='eza --all --long --header --icons --classify --git --no-user --tree --level=2'

# bat
export BAT_THEME="Dracula"

# fzf
[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh
export FZF_DEFAULT_COMMAND='fd --type f --hidden --exclude ".git"'
export FZF_DEFAULT_OPTS="--preview 'bat --color=always --style=header,grid --line-range :100 {}' --inline-info --height=70% --layout=reverse"

# Change directory
function chpwd() {el}

# Alias
alias o=open
alias a=touch
alias ..='z ..'

# uv venv
export VIRTUAL_ENV="/app/.venv"

# test
