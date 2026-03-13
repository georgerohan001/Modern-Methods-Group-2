import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  // The 'base' must match your GitHub repository name exactly
  base: '/Modern-Methods-Group-2/', 
  
  title: "Modern Methods: Group 1",
  description: "Final paper: Forest surveying using TLS, UAVs, and YOLOv11 deep learning.",
  
  themeConfig: {
    // Top Navigation Bar
    nav: [
      { text: 'Home', link: '/' },
      { text: 'The Report', link: '/motivation' },
    ],

    // The Sidebar Menu
    sidebar: [
      {
        text: 'I. Introduction',
        items: [
          { text: 'Project Motivation', link: '/motivation' },
          { text: 'Data Acquisition', link: '/data' },
        ]
      },
      {
        text: 'II. The Pipeline',
        items: [
          { text: 'Preprocessing & Slicing', link: '/preprocessing' },
          { text: 'Model Architecture', link: '/model' },
          { text: 'Training Process', link: '/training' },
        ]
      },
      {
        text: 'III. Outcomes',
        items: [
          { text: 'Results', link: '/results' },
          { text: 'Discussion', link: '/discussion' },
        ]
      }
    ],

    // Updated to reflect your actual repository link
    socialLinks: [
      { icon: 'github', link: 'https://github.com/georg/Modern-Methods-Group-2' }
    ],

    footer: {
      message: 'MSc Forestry: Modern Methods in TLS and UAV',
      copyright: 'Copyright © 2026 Group 1'
    }
  }
})