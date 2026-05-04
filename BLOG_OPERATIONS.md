# 博客常用操作

## 新建文章

```bash
hugo new posts/文章文件名.md
```

文件名建议用英文小写和短横线，例如：

```bash
hugo new posts/my-first-post.md
```

## 编辑文章

文章放在：

```text
content/posts/
```

打开对应的 `.md` 文件，正文写在开头的 `---` 下面。

发布前把：

```yaml
draft: true
```

改成：

```yaml
draft: false
```

## 添加图片

把图片放到：

```text
static/images/
```

文章里这样引用：

```markdown
![图片说明](/images/图片文件名.jpg)
```

## 本地预览

预览草稿和正式文章：

```bash
hugo server -D
```

打开终端里显示的网址，一般是：

```text
http://localhost:1313/
```

## 生成网站

```bash
hugo
```

## 常用文件位置

```text
hugo.yaml          网站配置
content/posts/     文章
static/images/     图片
themes/PaperMod/   主题，不建议改
```

