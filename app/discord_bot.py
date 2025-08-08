# ./app/discord_bot.py
import discord
import os
import logging
# --- 關鍵修正 1：導入 'log' 物件 ---
from .logger import log
# --- 關鍵修正 2：從 typing 模組導入 'List' ---
from typing import Dict, Any, Optional, List

class DiscordNotifier:
    def __init__(self):
        self.token = os.getenv("DISCORD_BOT_TOKEN")
        self.channel_id = int(os.getenv("DISCORD_CHANNEL_ID", 0))
        if not self.token or not self.channel_id:
            raise ValueError("DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID must be set in .env")
        
        intents = discord.Intents.default()
        self.client = discord.Client(intents=intents)
        self.channel = None

    async def connect(self):
        """連接到 Discord"""
        await self.client.login(self.token)
        self.channel = await self.client.fetch_channel(self.channel_id)
        if not self.channel:
            raise ConnectionError(f"Could not find Discord channel with ID {self.channel_id}")
        
    def _create_signal_embed(self, signal_data: Dict[str, Any], exchange_id: str, ai_prediction: Optional[Dict] = None) -> discord.Embed:
        """根據單一訊號數據，創建並返回一個 Embed 物件。"""
        signal_type = signal_data['signal_type']
        color = discord.Color.green() if signal_type == 'Long' else discord.Color.red()
        
        raw_symbol = signal_data['symbol']
        display_symbol = f"{raw_symbol.split(':')[0]} (Perpetual)"
        
        python_datetime = signal_data['test_trigger_time'].to_pydatetime()

        embed = discord.Embed(
            title=f"📈 {exchange_id.upper()} {signal_type} Signal: {display_symbol} ({signal_data['timeframe']})",
            description=f"A potential **{signal_type}** entry opportunity has been identified.",
            color=color,
            timestamp=python_datetime
        )
        
        embed.add_field(name="Trigger Price", value=f"`${signal_data['level_price']:.4f}`", inline=True)
        embed.add_field(name="Level Type", value=f"{signal_data['level_snr_type']} {signal_data['level_current_type']}", inline=True)
        
        embed.add_field(name="Signal Candle (Level Formed)", value=f"`{signal_data['signal_candle_time']}`", inline=False)
        embed.add_field(name="Flip Candle (Level Flipped)", value=f"`{signal_data['level_flipped_at']}`", inline=False)
        embed.add_field(name="Test Trigger (Latest Candle)", value=f"`{signal_data['test_trigger_time']}`", inline=False)

        if ai_prediction:
            pred_dir = ai_prediction['direction_pred']
            pred_mag = ai_prediction['magnitude_pred']
            ai_verdict = "✅ Confirmed" if (signal_type == 'Long' and pred_dir == 'Up') or \
                                          (signal_type == 'Short' and pred_dir == 'Down') else "⚠️ Contradicted"
            embed.add_field(name="🤖 AI Prediction", value=f"**Verdict:** {ai_verdict}\n**Predicted Direction:** {pred_dir}\n**Predicted Magnitude:** {pred_mag:.4f}%", inline=False)
        
        embed.set_footer(text="Crypto Quant Monitor")
        return embed

    async def send_signals_in_batch(self, signals_list: List[Dict[str, Any]], exchange_id: str):
        """接收一個訊號列表，並將它們作為多個 Embeds 在一則訊息中發送。"""
        if not self.channel:
            # 現在 'log' 已被正確導入
            log.error("Discord channel not initialized. Cannot send signals.")
            return

        embeds_to_send = []
        for signal_data in signals_list:
            if len(embeds_to_send) >= 10:
                log.warning("More than 10 signals found in a single run. Sending only the first 10.")
                break
            embed = self._create_signal_embed(signal_data, exchange_id)
            embeds_to_send.append(embed)
        
        if embeds_to_send:
            await self.channel.send(embeds=embeds_to_send)
            log.info(f"Successfully sent a batch of {len(embeds_to_send)} signals to Discord.")

    async def send_error(self, message: str):
        """發送錯誤/嚴重日誌"""
        if not self.channel:
            # 此處使用 print 是安全的，以防 logger 本身出錯
            print(f"CRITICAL: Discord channel not ready. Log message: {message}")
            return
        
        embed = discord.Embed(
            title="🚨 System Alert",
            description=message,
            color=discord.Color.dark_red()
        )
        await self.channel.send(embed=embed)

    async def close(self):
        """關閉 Discord 客戶端連接"""
        await self.client.close()


class DiscordLogHandler(logging.Handler):
    """一個將日誌記錄發送到 Discord 的自定義 Handler"""
    def __init__(self, notifier: DiscordNotifier):
        super().__init__()
        self.notifier = notifier

    def emit(self, record):
        import asyncio
        log_entry = self.format(record)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.notifier.send_error(f"```\n{log_entry}\n```"))
        except RuntimeError:
            asyncio.run(self.notifier.send_error(f"```\n{log_entry}\n```"))