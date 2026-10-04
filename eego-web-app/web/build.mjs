import { mkdir,copyFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
execFileSync(process.execPath,['--check','app.js'],{stdio:'inherit'});
await mkdir('dist',{recursive:true});
for(const file of ['index.html','app.js','style.css','icon.svg','manifest.webmanifest'])await copyFile(file,`dist/${file}`);
console.log('Eego static web build passed. Backend authentication is required separately.');
