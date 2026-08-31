---
layout: page
title: 标签
publish: false
---

<script setup lang="ts">
import { computed, ref } from 'vue'
import { data } from './.vitepress/theme/notes.data'

const active = ref('')

const tagStats = computed(() => {
  const map = new Map<string, number>()
  for (const note of data) {
    for (const tag of note.tags) {
      map.set(tag, (map.get(tag) || 0) + 1)
    }
  }
  return [...map.entries()].sort(
    (a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'zh')
  )
})

const list = computed(() =>
  active.value ? data.filter((n) => n.tags.includes(active.value)) : data
)

function toggle(tag: string) {
  active.value = active.value === tag ? '' : tag
}
</script>

<div class="tags-page">
  <div class="tag-cloud">
    <button class="tag-chip" :class="{ 'is-active': active === '' }" @click="active = ''">
      全部<span class="count">{{ data.length }}</span>
    </button>
    <button
      v-for="[tag, n] in tagStats"
      :key="tag"
      class="tag-chip"
      :class="{ 'is-active': active === tag }"
      @click="toggle(tag)"
    >{{ tag }}<span class="count">{{ n }}</span></button>
  </div>

  <p class="result-line">
    <template v-if="active">标签「{{ active }}」· {{ list.length }} 篇</template>
    <template v-else>共 {{ data.length }} 篇笔记 · {{ tagStats.length }} 个标签</template>
  </p>

  <ul class="note-list">
    <li v-for="note in list" :key="note.url">
      <a class="note-title" :href="note.url">{{ note.title }}</a>
      <span class="note-cat">{{ note.category }}</span>
    </li>
  </ul>
</div>

<style>
.tags-page {
  max-width: 780px;
  margin: 0 auto;
  padding: 24px 0 48px;
}
.tags-page .tag-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.tags-page .tag-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 999px;
  cursor: pointer;
  transition: color .2s, border-color .2s, background-color .2s;
}
.tags-page .tag-chip:hover {
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-brand-1);
}
.tags-page .tag-chip.is-active {
  color: var(--vp-c-bg);
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}
.tags-page .count {
  font-size: 12px;
  opacity: .65;
}
.tags-page .result-line {
  margin: 20px 0 12px;
  font-size: 13px;
  color: var(--vp-c-text-3);
}
.tags-page .note-list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.tags-page .note-list li {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 0;
  border-bottom: 1px solid var(--vp-c-divider);
}
.tags-page .note-title {
  font-size: 15px;
  color: var(--vp-c-text-1);
  text-decoration: none;
}
.tags-page .note-title:hover {
  color: var(--vp-c-brand-1);
}
.tags-page .note-cat {
  flex-shrink: 0;
  font-size: 12px;
  color: var(--vp-c-text-3);
}
</style>
