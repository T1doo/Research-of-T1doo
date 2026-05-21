---
title: "{{title}}"
authors: "{{authors}}"
year: "{% if date %}{{date | format('YYYY')}}{% endif %}"
journal: "{{publicationTitle}}"
doi: "{{DOI}}"
citekey: "{{citekey}}"
zotero-link: "{{desktopURI}}"
itemType: "{{itemType}}"
tags: [literature, T1D]
---

# {{title}}

> [!info] 元信息
> - **作者**：{{authors}}
> - **日期**：{% if date %}{{date | format("YYYY-MM-DD")}}{% endif %}
> - **来源**：{{publicationTitle}}
> - **DOI**：{% if DOI %}[{{DOI}}](https://doi.org/{{DOI}}){% endif %}
> - **Zotero**：[在 Zotero 中打开]({{desktopURI}})
> - **Citekey**：`@{{citekey}}`

## 📄 Abstract

{{abstractNote}}

## 🧠 我的思考
{% persist "my-thoughts" %}{% if isFirstImport %}

### 核心观点


### 方法论


### 与我研究的关联（T1D）


### 待解问题 / 后续追读

{% endif %}
{% endpersist %}

## ✏️ PDF 高亮与注释
{% persist "annotations" %}
{% if annotations and annotations.length > 0 %}
{% for a in annotations %}
### 📍 p.{{a.pageLabel}}{% if a.color %} ({{a.colorCategory}}){% endif %}

{% if a.annotatedText %}> {{a.annotatedText}}
{% endif %}
{% if a.imageRelativePath %}![[{{a.imageRelativePath}}]]
{% endif %}
{% if a.comment %}**💬 批注**：{{a.comment}}
{% endif %}

---
{% endfor %}
{% endif %}
{% endpersist %}
