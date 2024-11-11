FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

WORKDIR /app
COPY . .

RUN apt update && apt install -y python3 python3-pip vim zsh git curl bat fzf sqlite
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# set up python
RUN uv python install 3.12
# RUN apt install -y python3 python3-pip
# RUN pip install --upgrade pip
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv venv && uv pip install -r requirements.txt


# set up bat
RUN mkdir -p ~/.local/bin
RUN ln -s /usr/bin/batcat ~/.local/bin/bat

# install rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# install starship
RUN curl -sS https://starship.rs/install.sh | sh -s -- --yes

# install zoxide
RUN cargo install zoxide --locked

# install eza
RUN cargo install eza

RUN ln -s /app/.zshrc /root/.zshrc
RUN chsh -s /usr/bin/zsh