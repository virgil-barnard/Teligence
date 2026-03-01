# configs/ (user-specific, gitignored)

Put your secrets + identity here. These files are mounted into containers as **Compose secrets**
and appear at `/run/secrets/<name>` inside the container. 

- git/git.env: git user.name / user.email
- github/github.env: GitHub token + host
- glab/glab.env: GitLab token + host
- llm/llm.env: OpenAI key and (optional) base URL

IMPORTANT: Don't commit real tokens.
