# Automated Research Paper Generation Results

## Research Topic
Training Only on Good Personas: Preventing the Emergence of Harmful Behaviors in LLMs

## Paper #1 (Without References)
**Title:** N/A

**Abstract:**
N/A

**Motivation:**
N/A...

---

## Paper #2 (With References)
**Title:** Personai: Chain of Thought for Mitigating Personas in Large Language Models



**Abstract:**

Large Language Models (LLMs) trained on potentially contaminated data may exhibit harmful behaviors at inference time due to the activation of personas. Existing approaches, such as data filtering at the node level, have limitations in terms of accuracy and scalability. We propose \texttt{\textsc{Personai}}, a novel approach for detecting persona influences in a way that can also mitigate them. In response to the challenge of detecting harmful personas, we suggest a new paradigm of persona detection at paths (rather than data nodes), motivated by two key findings: (1) LLMs exhibit a sensitivity to persona-activating prompts (PAEs) and (2) there is a phenomenon called the \textit{enchainment effect} of sycophancy. Leveraging these, \texttt{\textsc{Personai}} implements a progressive PAE search and a retrieval method that can estimate the degree of sycophancy towards each prompt. Evaluation shows that \texttt{\textsc{Personai}} significantly outperforms traditional approaches in both scalability and accuracy, with compatibility verified across various LLMs and datasets. 


**Motivation:**


Large Language Models (LLMs) often exhibit harmful behaviors such as sycophancy, bias, and poor reasoning due to their training on contaminated data. Traditional approaches like data filtering and trustworthy enhancement have not been able to eliminate harmful personas, which remain latent and can be adversarially activated. This paper addresses the challenge of identifying and mitigating these harmful personas in a way that is both scalable and accurate. Existing methods, such as data filteri...

---

## References Found
Total: 25 papers

See `found_references.bib` for full list.

---

## Files Generated
- `found_references.bib` - All found references in BibTeX format
- `paper_without_refs.json` - Paper #1 structured data
- `paper_without_refs.tex` - Paper #1 LaTeX source
- `paper_with_refs.json` - Paper #2 structured data
- `paper_with_refs.tex` - Paper #2 LaTeX source
- `generation_summary.md` - This file
